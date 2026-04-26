use crate::{
    types::{ContentBlock, LLMRequest, LLMResponse, MessageContent, MessageRole, Usage},
    LLMProvider, StreamingCallback, StreamingChunk,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::StreamExt;
use rand::{distributions::Alphanumeric, Rng};
use regex::Regex;
use serde_json::Value;
use tokio::time::{sleep, timeout, Duration};
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

pub struct KiroNativeClient {
    client: reqwest::Client,
    model: String,
    base_url: String,
    auth_provider: crate::kiro_auth::KiroAuthProvider,
    profile_arn: Option<String>,
    auth_method: Option<String>,
}

impl KiroNativeClient {
    pub fn new(
        model: String,
        base_url: String,
        auth_provider: crate::kiro_auth::KiroAuthProvider,
        profile_arn: Option<String>,
        auth_method: Option<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            model,
            base_url,
            auth_provider,
            profile_arn,
            auth_method,
        }
    }

    fn generate_uuid_like() -> String {
        let random_hex: String = rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .filter(|c| c.is_ascii_hexdigit())
            .map(|c| c.to_ascii_lowercase())
            .take(32)
            .map(char::from)
            .collect();
        let mut hex = random_hex;
        while hex.len() < 32 {
            hex.push('0');
        }
        format!(
            "{}-{}-{}-{}-{}",
            &hex[0..8],
            &hex[8..12],
            &hex[12..16],
            &hex[16..20],
            &hex[20..32]
        )
    }

    fn normalize_model_for_kiro(model: &str) -> String {
        let mut name = model.trim().to_lowercase();
        if let Some(stripped) = name.strip_prefix("kiro-") {
            name = stripped.to_string();
        }
        if let Some(stripped) = name.strip_prefix("kr/") {
            name = stripped.to_string();
        }

        // claude-haiku-4-5[-suffix] -> claude-haiku-4.5
        let standard_re = Regex::new(
            r"^(claude-(?:haiku|sonnet|opus)-\d+)-(\d{1,2})(?:-(?:\d{8}|latest|\d+))?$",
        )
        .expect("valid regex");
        if let Some(caps) = standard_re.captures(&name) {
            return format!("{}.{}", &caps[1], &caps[2]);
        }

        // claude-3-7-sonnet[-suffix] -> claude-3.7-sonnet
        let legacy_re = Regex::new(
            r"^(claude)-(\d+)-(\d+)-(haiku|sonnet|opus)(?:-(?:\d{8}|latest|\d+))?$",
        )
        .expect("valid regex");
        if let Some(caps) = legacy_re.captures(&name) {
            return format!("{}-{}.{}-{}", &caps[1], &caps[2], &caps[3], &caps[4]);
        }

        name
    }

    fn extract_text(content: &MessageContent) -> String {
        match content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Structured(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.clone()),
                    ContentBlock::ToolResult { content, .. } => Some(content.to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    fn normalize_history(
        request: &LLMRequest,
        model: &str,
    ) -> (Vec<(MessageRole, String)>, (MessageRole, String)) {
        let mut messages: Vec<(MessageRole, String)> = request
            .messages
            .iter()
            .map(|m| (m.role.clone(), Self::extract_text(&m.content)))
            .collect();

        if messages.is_empty() {
            messages.push((MessageRole::User, "Hello".to_string()));
        }

        if !request.system_prompt.trim().is_empty() {
            if let Some((_, first)) = messages.iter_mut().find(|(r, _)| *r == MessageRole::User) {
                *first = format!("{}\n\n{}", request.system_prompt, first);
            } else {
                messages.insert(0, (MessageRole::User, request.system_prompt.clone()));
            }
        }

        if messages.first().is_some_and(|(role, _)| *role != MessageRole::User) {
            messages.insert(0, (MessageRole::User, "(empty)".to_string()));
        }

        let mut normalized: Vec<(MessageRole, String)> = Vec::new();
        for (role, content) in messages {
            if let Some((last_role, _)) = normalized.last() {
                if *last_role == role {
                    let synthetic_role = match role {
                        MessageRole::User => MessageRole::Assistant,
                        MessageRole::Assistant => MessageRole::User,
                    };
                    normalized.push((synthetic_role, "(empty)".to_string()));
                }
            }
            normalized.push((role, if content.is_empty() { "(empty)".to_string() } else { content }));
        }

        let current = normalized
            .pop()
            .unwrap_or((MessageRole::User, "Continue".to_string()));
        let mut history = normalized;

        // Kiro expects current message to be user input.
        let current = if current.0 == MessageRole::Assistant {
            history.push((MessageRole::Assistant, current.1));
            (MessageRole::User, "Continue".to_string())
        } else {
            current
        };

        // Keep model referenced to avoid unused warnings if future tweaks use it.
        let _ = model;
        (history, current)
    }

    fn build_payload(&self, request: &LLMRequest) -> Value {
        let model_id = Self::normalize_model_for_kiro(&self.model);
        let (history, current) = Self::normalize_history(request, &self.model);
        let history_json: Vec<Value> = history
            .into_iter()
            .map(|(role, content)| match role {
                MessageRole::User => serde_json::json!({
                    "userInputMessage": {
                        "content": content,
                        "modelId": model_id.clone(),
                        "origin": "AI_EDITOR"
                    }
                }),
                MessageRole::Assistant => serde_json::json!({
                    "assistantResponseMessage": {
                        "content": content
                    }
                }),
            })
            .collect();

        let mut payload = serde_json::json!({
            "conversationState": {
                "agentContinuationId": Self::generate_uuid_like(),
                "agentTaskType": "vibe",
                "chatTriggerType": "MANUAL",
                "conversationId": Self::generate_uuid_like(),
                "currentMessage": {
                    "userInputMessage": {
                        "content": if current.1.is_empty() { "Continue" } else { &current.1 },
                        "modelId": model_id,
                        "origin": "AI_EDITOR",
                        "userInputMessageContext": {
                            "tools": []
                        }
                    }
                },
                "history": []
            }
        });

        if !history_json.is_empty() {
            payload["conversationState"]["history"] = Value::Array(history_json);
        }
        if self.auth_method.as_deref() != Some("idc") {
            if let Some(profile_arn) = &self.profile_arn {
                if !profile_arn.trim().is_empty() {
                    payload["profileArn"] = Value::String(profile_arn.clone());
                }
            }
        }
        payload
    }
}

/// Known JSON event key prefixes for Kiro's AWS EventStream responses.
/// We anchor on these specific prefixes rather than a generic `{` search to
/// avoid being tricked by binary frame header bytes that may also contain `{`.
const KIRO_EVENT_PATTERNS: &[&str] = &[
    r#"{"content":"#,
    r#"{"name":"#,
    r#"{"input":"#,
    r#"{"stop":"#,
    r#"{"followupPrompt":"#,
    r#"{"assistantResponseMessage":"#,
    r#"{"delta":"#,
    r#"{"usage":"#,
    r#"{"contextUsagePercentage":"#,
];

fn find_matching_brace(text: &str, start_pos: usize) -> Option<usize> {
    if !text[start_pos..].starts_with('{') {
        return None;
    }
    let mut brace_count = 0i32;
    let mut in_string = false;
    let mut escape_next = false;
    for (idx, ch) in text.char_indices().skip(start_pos) {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            if ch == '{' {
                brace_count += 1;
            } else if ch == '}' {
                brace_count -= 1;
                if brace_count == 0 {
                    return Some(idx);
                }
            }
        }
    }
    None
}

/// Extract the next known Kiro JSON event from the buffer using pattern-anchored
/// prefix matching. Returns `None` if no complete event is available yet.
///
/// Instead of searching for the first bare `{` (which can match binary frame
/// header bytes in the AWS EventStream), we look for specific key prefixes like
/// `{"content":` that only appear at the start of real Kiro JSON payloads.
fn pop_next_kiro_event(buffer: &mut String) -> Option<String> {
    // Find the earliest occurrence of any known event prefix.
    let earliest = KIRO_EVENT_PATTERNS
        .iter()
        .filter_map(|pat| buffer.find(pat).map(|pos| (pos, *pat)))
        .min_by_key(|(pos, _)| *pos);

    let (start, _pat) = earliest?;
    // IMPORTANT: pass a substring slice starting at `start` with offset 0.
    // find_matching_brace uses .skip(start_pos) which counts *chars*, not bytes.
    // The buffer contains \u{FFFD} replacement chars from binary AWS EventStream
    // frame bytes (3 bytes each, 1 char each), so byte offset != char count.
    // By slicing first we ensure the scan always begins at the correct `{`.
    let end_in_sub = find_matching_brace(&buffer[start..], 0)?;
    let end = start + end_in_sub;
    let json_str = buffer[start..=end].to_string();
    *buffer = buffer[end + 1..].to_string();
    Some(json_str)
}

fn process_kiro_event(
    event: &Value,
    last_content: &mut Option<String>,
    full_text: &mut String,
    current_tool_call: &mut Option<PendingToolCall>,
    tool_calls: &mut Vec<PendingToolCall>,
    seen_event_count: &mut usize,
    seen_event_types: &mut Vec<String>,
    seen_event_key_samples: &mut Vec<String>,
    streaming_callback: Option<&StreamingCallback>,
) -> Result<()> {
    *seen_event_count += 1;

    if let Some(obj) = event.as_object() {
        let keys = obj.keys().cloned().collect::<Vec<_>>().join(",");
        if !keys.is_empty() && !seen_event_key_samples.iter().any(|k| k == &keys) {
            seen_event_key_samples.push(keys);
        }
    }

    let mut event_type = "other".to_string();
    if event.get("content").is_some() {
        event_type = "content".to_string();
    } else if event.get("followupPrompt").is_some() {
        event_type = "followup".to_string();
    } else if event.get("assistantResponseMessage").is_some() {
        event_type = "assistant_response".to_string();
    } else if event.get("delta").is_some() {
        event_type = "delta".to_string();
    } else if event.get("name").is_some() {
        event_type = "tool_start".to_string();
    } else if event.get("input").is_some() {
        event_type = "tool_input".to_string();
    } else if event.get("stop").is_some() {
        event_type = "tool_stop".to_string();
    } else if event.get("usage").is_some() {
        event_type = "usage".to_string();
    } else if event.get("contextUsagePercentage").is_some() {
        event_type = "context_usage".to_string();
    }
    if !seen_event_types.iter().any(|t| t == &event_type) {
        seen_event_types.push(event_type.clone());
    }

    let mut text_candidates: Vec<String> = Vec::new();
    if let Some(s) = event.get("content").and_then(|v| v.as_str()) {
        text_candidates.push(s.to_string());
    }
    if let Some(s) = event.get("followupPrompt").and_then(|v| v.as_str()) {
        text_candidates.push(s.to_string());
    }
    if let Some(s) = event
        .get("assistantResponseMessage")
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
    {
        text_candidates.push(s.to_string());
    }
    if let Some(s) = event
        .get("delta")
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
    {
        text_candidates.push(s.to_string());
    }
    // Some Kiro events put text in nested payload.data.content style
    if let Some(s) = event
        .pointer("/payload/data/content")
        .and_then(|v| v.as_str())
    {
        text_candidates.push(s.to_string());
    }

    for content in text_candidates {
        if content.is_empty() || last_content.as_ref() == Some(&content) {
            continue;
        }
        *last_content = Some(content.clone());
        full_text.push_str(&content);
        if let Some(cb) = streaming_callback {
            cb(&StreamingChunk::Text(content))?;
        }
    }

    if event_type == "tool_start" {
        if let Some(tc) = current_tool_call.take() {
            tool_calls.push(tc);
        }
        let name = event
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let id = event
            .get("toolUseId")
            .and_then(|v| v.as_str())
            .unwrap_or("tool_call")
            .to_string();
        let arguments = match event.get("input") {
            Some(Value::String(s)) => s.clone(),
            Some(v) => serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string()),
            None => String::new(),
        };
        *current_tool_call = Some(PendingToolCall {
            id,
            name,
            arguments,
        });
    } else if event_type == "tool_input" {
        if let Some(tc) = current_tool_call.as_mut() {
            match event.get("input") {
                Some(Value::String(s)) => tc.arguments.push_str(s),
                Some(v) => {
                    tc.arguments
                        .push_str(&serde_json::to_string(v).unwrap_or_else(|_| String::new()));
                }
                None => {}
            }
        }
    }

    if event_type == "tool_stop"
        || (event_type == "tool_start"
            && event.get("stop").and_then(|v| v.as_bool()).unwrap_or(false))
    {
        if let Some(tc) = current_tool_call.take() {
            tool_calls.push(tc);
        }
    }

    Ok(())
}

#[async_trait]
impl LLMProvider for KiroNativeClient {
    async fn send_message(
        &mut self,
        request: LLMRequest,
        streaming_callback: Option<&StreamingCallback>,
    ) -> Result<LLMResponse> {
        const MAX_EMPTY_RETRIES: usize = 3;
        const EMPTY_RETRY_DELAY_SECS: u64 = 2;

        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 1..=MAX_EMPTY_RETRIES {
            match self
                .send_message_once(&request, streaming_callback)
                .await
            {
                Ok(response) => return Ok(response),
                Err(e) if e.to_string().contains("Kiro API returned no textual content") => {
                    warn!(
                        "Kiro returned no textual content (attempt {}/{}), retrying in {}s",
                        attempt, MAX_EMPTY_RETRIES, EMPTY_RETRY_DELAY_SECS
                    );
                    last_err = Some(e);
                    if attempt < MAX_EMPTY_RETRIES {
                        sleep(Duration::from_secs(EMPTY_RETRY_DELAY_SECS)).await;
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("Kiro API returned no textual content")))
    }
}

impl KiroNativeClient {
    async fn send_message_once(
        &self,
        request: &LLMRequest,
        streaming_callback: Option<&StreamingCallback>,
    ) -> Result<LLMResponse> {
        let payload = self.build_payload(request);
        let url = format!("{}/generateAssistantResponse", self.base_url.trim_end_matches('/'));
        debug!(
            "Kiro request: url={}, model={}, auth_method={:?}, has_profile_arn={}, payload_bytes={}",
            url,
            self.model,
            self.auth_method,
            self.profile_arn.as_ref().is_some_and(|s| !s.is_empty()),
            serde_json::to_vec(&payload).map(|v| v.len()).unwrap_or(0)
        );

        let auth_headers =
            crate::anthropic::AuthProvider::get_auth_headers(&self.auth_provider).await?;

        let mut req = self.client.post(&url).json(&payload);
        for (k, v) in auth_headers {
            req = req.header(&k, &v);
        }
        req = req.header("Content-Type", "application/json");

        let response = req.send().await.context("Failed to call Kiro API")?;
        let status = response.status();
        info!("Kiro response status: {}", status);
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Kiro API error {}: {}", status, body);
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut full_text = String::new();
        let mut last_content: Option<String> = None;
        let mut current_tool_call: Option<PendingToolCall> = None;
        let mut tool_calls: Vec<PendingToolCall> = Vec::new();
        let mut seen_event_count: usize = 0;
        let mut seen_event_types: Vec<String> = Vec::new();
        let mut seen_event_key_samples: Vec<String> = Vec::new();

        let first_chunk = timeout(Duration::from_secs(25), stream.next())
            .await
            .map_err(|_| anyhow::anyhow!("Kiro API timeout: no first token within 25s"))?;
        let Some(first_chunk) = first_chunk else {
            anyhow::bail!("Kiro API returned empty stream");
        };
        let first_chunk = first_chunk.context("Failed reading first Kiro stream chunk")?;
        debug!(
            "Kiro first chunk bytes={}, preview={}",
            first_chunk.len(),
            String::from_utf8_lossy(&first_chunk)
                .chars()
                .take(240)
                .collect::<String>()
        );
        buffer.push_str(&String::from_utf8_lossy(&first_chunk));
        while let Some(json_str) = pop_next_kiro_event(&mut buffer) {
            let Ok(event) = serde_json::from_str::<Value>(&json_str) else {
                debug!(
                    "Kiro event parse failed (first chunk path), raw={}",
                    json_str.chars().take(240).collect::<String>()
                );
                continue;
            };
            process_kiro_event(
                &event,
                &mut last_content,
                &mut full_text,
                &mut current_tool_call,
                &mut tool_calls,
                &mut seen_event_count,
                &mut seen_event_types,
                &mut seen_event_key_samples,
                streaming_callback,
            )?;
        }

        loop {
            let next_chunk = timeout(Duration::from_secs(30), stream.next())
                .await
                .map_err(|_| anyhow::anyhow!("Kiro API timeout: no stream activity within 30s"))?;
            let Some(next_chunk) = next_chunk else {
                break;
            };
            let chunk = next_chunk.context("Failed reading Kiro stream")?;
            debug!(
                "Kiro stream chunk bytes={}, preview={}",
                chunk.len(),
                String::from_utf8_lossy(&chunk)
                    .chars()
                    .take(180)
                    .collect::<String>()
            );
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(json_str) = pop_next_kiro_event(&mut buffer) {
                let Ok(event) = serde_json::from_str::<Value>(&json_str) else {
                    debug!(
                        "Kiro event parse failed, raw={}",
                        json_str.chars().take(240).collect::<String>()
                    );
                    continue;
                };
                process_kiro_event(
                    &event,
                    &mut last_content,
                    &mut full_text,
                    &mut current_tool_call,
                    &mut tool_calls,
                    &mut seen_event_count,
                    &mut seen_event_types,
                    &mut seen_event_key_samples,
                    streaming_callback,
                )?;
            }
        }

        if let Some(tc) = current_tool_call.take() {
            tool_calls.push(tc);
        }

        if full_text.trim().is_empty() && tool_calls.is_empty() {
            warn!(
                "Kiro returned no text/tool events. seen_event_count={}, event_types={:?}, buffer_tail={}",
                seen_event_count,
                seen_event_types,
                buffer.chars().rev().take(240).collect::<String>().chars().rev().collect::<String>()
            );
            warn!("Kiro seen event key samples: {:?}", seen_event_key_samples);
            anyhow::bail!("Kiro API returned no textual content");
        }

        if let Some(cb) = streaming_callback {
            cb(&StreamingChunk::StreamingComplete)?;
        }

        let mut blocks = Vec::new();
        if !full_text.is_empty() {
            blocks.push(ContentBlock::new_text(full_text));
        }
        for tc in tool_calls {
            let args_json: Value =
                serde_json::from_str(&tc.arguments).unwrap_or_else(|_| serde_json::json!({}));
            blocks.push(ContentBlock::new_tool_use(tc.id, tc.name, args_json));
        }

        Ok(LLMResponse {
            content: blocks,
            usage: Usage::zero(),
            rate_limit_info: None,
        })
    }
}
