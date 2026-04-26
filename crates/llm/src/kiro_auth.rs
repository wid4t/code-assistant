//! Kiro OAuth authentication module.
//!
//! Implements the AWS Builder ID / IAM Identity Center device authorization
//! flow used by recent Kiro clients. Tokens are persisted in `providers.json`
//! under `config.kiro_tokens`.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use tracing::{info, warn};

// Re-export for consumers
pub use crate::openai_responses::{AuthProvider, RequestCustomizer};

/// Kiro refresh grant type.
const REFRESH_TOKEN_GRANT_TYPE: &str = "refresh_token";

/// Responses WebSocket beta header value.
const RESPONSES_WS_BETA_HEADER: &str = "responses_websockets=2026-02-06";
/// Responses HTTP/SSE beta header value.
const RESPONSES_HTTP_BETA_HEADER: &str = "responses=experimental";

/// Responses endpoint suffix.
const RESPONSES_PATH: &str = "/responses";

/// Default auth method used by Builder ID flow.
const IDC_AUTH_METHOD: &str = "idc";

/// Default label for providers.json bootstrap entries.
const DEFAULT_PROVIDER_LABEL: &str = "Kiro OAuth (Builder ID)";

/// Supported provider kind for Kiro auth entries.
const DEFAULT_PROVIDER_KIND: &str = "kiro";

/// Older provider kind used by previous Kiro auth bootstrap logic.
const LEGACY_PROVIDER_KIND: &str = "openai-responses-ws";
const LEGACY_PROVIDER_KIND_OPENAI_COMPATIBLE: &str = "openai-compatible";
const LEGACY_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// Message used when auth tokens are not available in config.
pub const LOGIN_HINT_MESSAGE: &str =
    "No Kiro auth tokens found in provider config. Run `code-assistant kiro-login` first.";

/// Message used when required refresh metadata is missing.
const REFRESH_METADATA_MISSING_MESSAGE: &str =
    "Kiro auth state is missing refresh metadata. Run `code-assistant kiro-login` again.";

/// Message used when refresh token itself is missing.
const REFRESH_TOKEN_MISSING_MESSAGE: &str =
    "Kiro auth state does not include a refresh token. Run `code-assistant kiro-login` again.";

/// Message used when refresh endpoint returns a terminal auth failure.
const REFRESH_REAUTH_MESSAGE: &str =
    "Kiro refresh token is invalid or expired. Run `code-assistant kiro-login` again.";

/// Default base URL for Kiro Responses APIs.
pub const DEFAULT_BASE_URL: &str = "https://q.us-east-1.amazonaws.com";

/// Default provider id for persisted Kiro tokens.
pub const DEFAULT_PROVIDER_ID: &str = "claude-kiro-oauth";

/// Default AWS region for Builder ID auth.
pub const DEFAULT_REGION: &str = "us-east-1";

/// Default AWS Builder ID start URL.
pub const BUILDER_ID_START_URL: &str = "https://view.awsapps.com/start";

/// User agent expected by the Kiro IDC flow.
const KIRO_USER_AGENT: &str = "KiroIDE";

/// Time window for considering a token close to expiry.
const EXPIRY_GRACE_MINUTES: i64 = 5;

/// Device token polling grant type.
const DEVICE_CODE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";

/// Scopes requested from AWS Builder ID.
const KIRO_SCOPES: [&str; 5] = [
    "codewhisperer:completions",
    "codewhisperer:analysis",
    "codewhisperer:conversations",
    "codewhisperer:transformations",
    "codewhisperer:taskassist",
];

/// Kiro OAuth tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KiroTokens {
    pub access_token: String,
    pub refresh_token: String,
    pub profile_arn: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub region: Option<String>,
    pub auth_method: Option<String>,
}

/// Persisted Kiro auth state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KiroAuthState {
    pub tokens: KiroTokens,
    pub last_refresh: DateTime<Utc>,
}

impl KiroAuthState {
    pub fn needs_refresh(&self) -> bool {
        match self.tokens.expires_at {
            Some(expires_at) => expires_at <= Utc::now() + Duration::minutes(EXPIRY_GRACE_MINUTES),
            None => false,
        }
    }
}

/// Pluggable storage backend for Kiro OAuth tokens.
pub trait KiroTokenStorage: Send + Sync {
    fn save(&self, state: &KiroAuthState) -> Result<()>;
    fn load(&self) -> Result<Option<KiroAuthState>>;
    fn delete(&self) -> Result<()>;
}

/// providers.json backed token storage.
pub struct ProvidersJsonTokenStorage {
    provider_id: String,
    providers_path: Option<PathBuf>,
}

impl ProvidersJsonTokenStorage {
    pub fn new(provider_id: String, providers_path: Option<PathBuf>) -> Self {
        Self {
            provider_id,
            providers_path,
        }
    }
}

impl KiroTokenStorage for ProvidersJsonTokenStorage {
    fn save(&self, state: &KiroAuthState) -> Result<()> {
        save_auth_state_to_provider(state, &self.provider_id, self.providers_path.as_deref())
    }

    fn load(&self) -> Result<Option<KiroAuthState>> {
        load_auth_state_from_provider(&self.provider_id, self.providers_path.as_deref())
    }

    fn delete(&self) -> Result<()> {
        delete_auth_state_from_provider(&self.provider_id, self.providers_path.as_deref())
    }
}

/// Build the Kiro Responses base URL for a given AWS region.
pub fn default_base_url_for_region(region: Option<&str>) -> String {
    let region = region
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_REGION);
    format!("https://q.{region}.amazonaws.com")
}

fn save_auth_state_to_provider(
    state: &KiroAuthState,
    provider_id: &str,
    providers_path: Option<&Path>,
) -> Result<()> {
    let tokens_value = serde_json::json!({
        "access_token": state.tokens.access_token,
        "refresh_token": state.tokens.refresh_token,
        "profile_arn": state.tokens.profile_arn,
        "expires_at": state.tokens.expires_at,
        "client_id": state.tokens.client_id,
        "client_secret": state.tokens.client_secret,
        "region": state.tokens.region,
        "auth_method": state.tokens.auth_method,
        "last_refresh": state.last_refresh,
    });

    crate::provider_config::ConfigurationSystem::save_providers_config(providers_path, |raw| {
        let obj = raw
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("providers.json is not a JSON object"))?;

        let provider_entry = obj.entry(provider_id.to_string()).or_insert_with(|| {
            serde_json::json!({
                "label": DEFAULT_PROVIDER_LABEL,
                "provider": DEFAULT_PROVIDER_KIND,
                "config": { "kiro_auth": true }
            })
        });

        // Migrate legacy provider kinds so older entries keep working.
        if provider_entry
            .get("provider")
            .and_then(|v| v.as_str())
            .is_some_and(|kind| {
                kind == LEGACY_PROVIDER_KIND || kind == LEGACY_PROVIDER_KIND_OPENAI_COMPATIBLE
            })
        {
            provider_entry["provider"] = serde_json::Value::String(DEFAULT_PROVIDER_KIND.to_string());
        }

        if provider_entry.get("label").is_none() {
            provider_entry["label"] = serde_json::Value::String(DEFAULT_PROVIDER_LABEL.to_string());
        }

        if provider_entry.get("provider").is_none() {
            provider_entry["provider"] =
                serde_json::Value::String(DEFAULT_PROVIDER_KIND.to_string());
        }

        if provider_entry.get("config").is_none() {
            provider_entry["config"] = serde_json::json!({});
        }

        let provider_obj = provider_entry
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("provider entry is not a JSON object"))?;

        let config = provider_obj
            .entry("config".to_string())
            .or_insert_with(|| serde_json::json!({}));

        if !config.is_object() {
            *config = serde_json::json!({});
        }

        let config_obj = config
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("provider config is not a JSON object"))?;

        config_obj.insert("kiro_auth".to_string(), serde_json::Value::Bool(true));
        config_obj.insert("kiro_tokens".to_string(), tokens_value);

        let derived_base_url = default_base_url_for_region(state.tokens.region.as_deref());
        let needs_base_url_update = config_obj
            .get("base_url")
            .and_then(|v| v.as_str())
            .is_none_or(|value| value == LEGACY_OPENAI_BASE_URL);
        if needs_base_url_update {
            config_obj.insert(
                "base_url".to_string(),
                serde_json::Value::String(derived_base_url),
            );
        }

        Ok(())
    })?;

    info!(
        "Saved Kiro auth tokens to providers.json (provider: {})",
        provider_id
    );
    Ok(())
}

fn load_auth_state_from_provider(
    provider_id: &str,
    providers_path: Option<&Path>,
) -> Result<Option<KiroAuthState>> {
    let path = if let Some(p) = providers_path {
        p.to_path_buf()
    } else {
        crate::provider_config::ConfigurationSystem::default_providers_path()
    };

    if !path.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&path)?;
    let raw: serde_json::Value = serde_json::from_str(&content)?;

    let tokens_value = raw
        .get(provider_id)
        .and_then(|p| p.get("config"))
        .and_then(|c| c.get("kiro_tokens"));

    match tokens_value {
        Some(tv) => {
            let access_token = tv
                .get("access_token")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let refresh_token = tv
                .get("refresh_token")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            if access_token.is_empty() {
                return Ok(None);
            }

            let profile_arn = tv
                .get("profile_arn")
                .and_then(|v| v.as_str())
                .map(String::from);
            let expires_at = tv
                .get("expires_at")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok());
            let client_id = tv
                .get("client_id")
                .and_then(|v| v.as_str())
                .map(String::from);
            let client_secret = tv
                .get("client_secret")
                .and_then(|v| v.as_str())
                .map(String::from);
            let region = tv.get("region").and_then(|v| v.as_str()).map(String::from);
            let auth_method = tv
                .get("auth_method")
                .and_then(|v| v.as_str())
                .map(String::from);
            let last_refresh: DateTime<Utc> = tv
                .get("last_refresh")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(Utc::now);

            Ok(Some(KiroAuthState {
                tokens: KiroTokens {
                    access_token,
                    refresh_token,
                    profile_arn,
                    expires_at,
                    client_id,
                    client_secret,
                    region,
                    auth_method,
                },
                last_refresh,
            }))
        }
        None => Ok(None),
    }
}

fn delete_auth_state_from_provider(provider_id: &str, providers_path: Option<&Path>) -> Result<()> {
    crate::provider_config::ConfigurationSystem::save_providers_config(providers_path, |raw| {
        if let Some(provider) = raw.get_mut(provider_id) {
            if let Some(config) = provider.get_mut("config") {
                if let Some(obj) = config.as_object_mut() {
                    obj.remove("kiro_tokens");
                }
            }
        }
        Ok(())
    })?;

    info!(
        "Removed Kiro auth tokens from providers.json (provider: {})",
        provider_id
    );
    Ok(())
}

/// Load auth state from the provider config value (already parsed by the factory).
pub fn load_auth_state_from_config(config: &serde_json::Value) -> Option<KiroAuthState> {
    let tv = config.get("kiro_tokens")?;

    let access_token = tv
        .get("access_token")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    if access_token.is_empty() {
        return None;
    }

    let refresh_token = tv
        .get("refresh_token")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let profile_arn = tv
        .get("profile_arn")
        .and_then(|v| v.as_str())
        .map(String::from);
    let expires_at = tv
        .get("expires_at")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse().ok());
    let client_id = tv
        .get("client_id")
        .and_then(|v| v.as_str())
        .map(String::from);
    let client_secret = tv
        .get("client_secret")
        .and_then(|v| v.as_str())
        .map(String::from);
    let region = tv.get("region").and_then(|v| v.as_str()).map(String::from);
    let auth_method = tv
        .get("auth_method")
        .and_then(|v| v.as_str())
        .map(String::from);
    let last_refresh: DateTime<Utc> = tv
        .get("last_refresh")
        .and_then(|v| v.as_str())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(Utc::now);

    Some(KiroAuthState {
        tokens: KiroTokens {
            access_token,
            refresh_token,
            profile_arn,
            expires_at,
            client_id,
            client_secret,
            region,
            auth_method,
        },
        last_refresh,
    })
}

/// Refresh Kiro access tokens using the persisted Builder ID refresh token.
pub async fn refresh_tokens(auth_state: &KiroAuthState) -> Result<KiroAuthState> {
    let refresh_token = if auth_state.tokens.refresh_token.is_empty() {
        bail!(REFRESH_TOKEN_MISSING_MESSAGE);
    } else {
        auth_state.tokens.refresh_token.clone()
    };

    let client_id = auth_state
        .tokens
        .client_id
        .clone()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!(REFRESH_METADATA_MISSING_MESSAGE))?;
    let client_secret = auth_state
        .tokens
        .client_secret
        .clone()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!(REFRESH_METADATA_MISSING_MESSAGE))?;

    let region = auth_state
        .tokens
        .region
        .clone()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_REGION.to_string());
    let oidc_endpoint = build_sso_oidc_endpoint(&region);

    let response = reqwest::Client::new()
        .post(format!("{oidc_endpoint}/token"))
        .header("Content-Type", "application/json")
        .header("User-Agent", KIRO_USER_AGENT)
        .json(&serde_json::json!({
            "clientId": client_id,
            "clientSecret": client_secret,
            "grantType": REFRESH_TOKEN_GRANT_TYPE,
            "refreshToken": refresh_token,
        }))
        .send()
        .await
        .context("Failed to send Kiro refresh request")?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    let payload = if body.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str::<Value>(&body)
            .with_context(|| format!("Invalid JSON from Kiro refresh endpoint: {body}"))?
    };

    let parsed = parse_token_payload(&payload);
    if let Some(error) = parsed.error.as_deref() {
        let details = parsed.error_description.unwrap_or_default();
        match error {
            "invalid_grant" | "invalid_client" | "expired_token" => {
                bail!("{REFRESH_REAUTH_MESSAGE}")
            }
            _ => bail!("Kiro token refresh failed: {} {}", error, details),
        }
    }

    if !status.is_success() {
        bail!("Kiro token refresh failed with status {}: {}", status, body);
    }

    let access_token = parsed
        .access_token
        .ok_or_else(|| anyhow::anyhow!("Kiro refresh response missing access token"))?;
    let refreshed_refresh_token = parsed
        .refresh_token
        .unwrap_or_else(|| auth_state.tokens.refresh_token.clone());
    let expires_at = parsed
        .expires_in
        .map(|seconds| Utc::now() + Duration::seconds(seconds.max(1)));

    Ok(KiroAuthState {
        tokens: KiroTokens {
            access_token,
            refresh_token: refreshed_refresh_token,
            profile_arn: auth_state.tokens.profile_arn.clone(),
            expires_at,
            client_id: Some(client_id),
            client_secret: Some(client_secret),
            region: Some(region),
            auth_method: auth_state
                .tokens
                .auth_method
                .clone()
                .or_else(|| Some(IDC_AUTH_METHOD.to_string())),
        },
        last_refresh: Utc::now(),
    })
}

/// Manages Kiro auth tokens and refreshes them on demand for Responses requests.
pub struct KiroAuthProvider {
    state: Arc<RwLock<KiroAuthState>>,
    storage: Arc<dyn KiroTokenStorage>,
}

impl KiroAuthProvider {
    pub fn new(state: KiroAuthState, storage: Arc<dyn KiroTokenStorage>) -> Self {
        Self {
            state: Arc::new(RwLock::new(state)),
            storage,
        }
    }

    async fn ensure_fresh_tokens(&self) -> Result<()> {
        let needs_refresh = {
            let state = self.state.read().unwrap();
            state.needs_refresh()
        };

        if !needs_refresh {
            return Ok(());
        }

        let current_state = self.state.read().unwrap().clone();
        let refreshed_state = refresh_tokens(&current_state).await?;
        self.storage.save(&refreshed_state)?;

        let mut state = self.state.write().unwrap();
        *state = refreshed_state;
        info!("Successfully refreshed Kiro auth tokens");

        Ok(())
    }
}

#[async_trait]
impl AuthProvider for KiroAuthProvider {
    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>> {
        self.ensure_fresh_tokens().await?;

        let state = self.state.read().unwrap();
        Ok(vec![
            (
                "Authorization".to_string(),
                format!("Bearer {}", state.tokens.access_token),
            ),
            (
                "x-amz-sso_bearer_token".to_string(),
                state.tokens.access_token.clone(),
            ),
        ])
    }
}

#[async_trait]
impl crate::openai::AuthProvider for KiroAuthProvider {
    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>> {
        self.ensure_fresh_tokens().await?;

        let state = self.state.read().unwrap();
        Ok(vec![
            (
                "Authorization".to_string(),
                format!("Bearer {}", state.tokens.access_token),
            ),
            (
                "x-amz-sso_bearer_token".to_string(),
                state.tokens.access_token.clone(),
            ),
        ])
    }
}

#[async_trait]
impl crate::anthropic::AuthProvider for KiroAuthProvider {
    async fn get_auth_headers(&self) -> Result<Vec<(String, String)>> {
        self.ensure_fresh_tokens().await?;

        let state = self.state.read().unwrap();
        Ok(vec![
            (
                "Authorization".to_string(),
                format!("Bearer {}", state.tokens.access_token),
            ),
            (
                "x-amz-sso_bearer_token".to_string(),
                state.tokens.access_token.clone(),
            ),
        ])
    }
}

/// Request customizer for Kiro/OpenAI-compatible WebSocket Responses requests.
pub struct KiroWsRequestCustomizer;

impl RequestCustomizer for KiroWsRequestCustomizer {
    fn customize_request(&self, _request: &mut serde_json::Value) -> Result<()> {
        Ok(())
    }

    fn get_additional_headers(&self) -> Vec<(String, String)> {
        vec![(
            "OpenAI-Beta".to_string(),
            RESPONSES_WS_BETA_HEADER.to_string(),
        )]
    }

    fn customize_url(&self, base_url: &str, _streaming: bool) -> String {
        format!("{base_url}{RESPONSES_PATH}")
    }
}

/// Create an `OpenAIResponsesWsClient` configured for Kiro Builder ID auth.
pub fn create_kiro_responses_ws_client(
    auth_state: KiroAuthState,
    model: String,
    base_url: String,
    storage: Arc<dyn KiroTokenStorage>,
) -> crate::openai_responses_ws::OpenAIResponsesWsClient {
    let auth_provider = Box::new(KiroAuthProvider::new(auth_state, storage));
    let request_customizer = Box::new(KiroWsRequestCustomizer);

    crate::openai_responses_ws::OpenAIResponsesWsClient::with_customization(
        model,
        base_url,
        auth_provider,
        request_customizer,
    )
}

/// Request customizer for Kiro HTTP/SSE Responses requests.
pub struct KiroRequestCustomizer;

impl RequestCustomizer for KiroRequestCustomizer {
    fn customize_request(&self, _request: &mut serde_json::Value) -> Result<()> {
        Ok(())
    }

    fn get_additional_headers(&self) -> Vec<(String, String)> {
        vec![(
            "OpenAI-Beta".to_string(),
            RESPONSES_HTTP_BETA_HEADER.to_string(),
        )]
    }

    fn customize_url(&self, base_url: &str, _streaming: bool) -> String {
        format!("{base_url}{RESPONSES_PATH}")
    }
}

/// Create an `OpenAIResponsesClient` configured for Kiro Builder ID auth.
pub fn create_kiro_responses_client(
    auth_state: KiroAuthState,
    model: String,
    base_url: String,
    storage: Arc<dyn KiroTokenStorage>,
) -> crate::openai_responses::OpenAIResponsesClient {
    let auth_provider = Box::new(KiroAuthProvider::new(auth_state, storage));
    let request_customizer = Box::new(KiroRequestCustomizer);

    crate::openai_responses::OpenAIResponsesClient::with_customization(
        model,
        base_url,
        auth_provider,
        request_customizer,
    )
}

/// Request customizer for Kiro OpenAI-compatible chat/completions requests.
pub struct KiroOpenAIRequestCustomizer;

impl crate::openai::RequestCustomizer for KiroOpenAIRequestCustomizer {
    fn customize_request(&self, _request: &mut serde_json::Value) -> Result<()> {
        Ok(())
    }

    fn get_additional_headers(&self) -> Vec<(String, String)> {
        vec![("Content-Type".to_string(), "application/json".to_string())]
    }

    fn customize_url(&self, base_url: &str, _streaming: bool) -> String {
        format!("{base_url}/chat/completions")
    }
}

/// Create an `OpenAIClient` configured for Kiro Builder ID auth over HTTPS.
pub fn create_kiro_openai_client(
    auth_state: KiroAuthState,
    model: String,
    base_url: String,
    storage: Arc<dyn KiroTokenStorage>,
) -> crate::openai::OpenAIClient {
    let auth_provider = Box::new(KiroAuthProvider::new(auth_state, storage));
    let request_customizer = Box::new(KiroOpenAIRequestCustomizer);

    crate::openai::OpenAIClient::with_customization(
        model,
        base_url,
        auth_provider,
        request_customizer,
    )
}

/// Request customizer for Kiro Anthropic-compatible messages requests.
pub struct KiroAnthropicRequestCustomizer;

impl crate::anthropic::RequestCustomizer for KiroAnthropicRequestCustomizer {
    fn customize_request(&self, _request: &mut serde_json::Value) -> Result<()> {
        Ok(())
    }

    fn get_additional_headers(&self) -> Vec<(String, String)> {
        vec![("anthropic-version".to_string(), "2023-06-01".to_string())]
    }

    fn customize_url(&self, base_url: &str, _streaming: bool) -> String {
        format!("{base_url}/messages")
    }
}

/// Create an `AnthropicClient` configured for Kiro Builder ID auth over HTTPS.
pub fn create_kiro_anthropic_client(
    auth_state: KiroAuthState,
    model: String,
    base_url: String,
    storage: Arc<dyn KiroTokenStorage>,
) -> crate::anthropic::AnthropicClient {
    let auth_provider = Box::new(KiroAuthProvider::new(auth_state, storage));
    let request_customizer = Box::new(KiroAnthropicRequestCustomizer);
    let message_converter = Box::new(crate::anthropic::DefaultMessageConverter::new());

    crate::anthropic::AnthropicClient::with_customization(
        model,
        base_url,
        auth_provider,
        request_customizer,
        message_converter,
    )
}

#[derive(Debug, Clone)]
pub struct LoginResult {
    pub auth_state: KiroAuthState,
}

#[derive(Debug, Clone)]
pub struct DeviceAuthorization {
    pub verification_url: String,
    pub verification_uri_complete: String,
    pub user_code: String,
    pub interval_seconds: u64,
    pub expires_in_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KiroAuthStatus {
    pub authenticated: bool,
    pub expires_at: Option<DateTime<Utc>>,
    pub needs_refresh: bool,
}

pub fn get_auth_status(storage: &dyn KiroTokenStorage) -> KiroAuthStatus {
    match storage.load() {
        Ok(Some(state)) => KiroAuthStatus {
            authenticated: true,
            expires_at: state.tokens.expires_at,
            needs_refresh: state.needs_refresh(),
        },
        _ => KiroAuthStatus {
            authenticated: false,
            expires_at: None,
            needs_refresh: false,
        },
    }
}

/// Start Kiro Builder ID device-code login flow.
pub async fn start_login_flow(
    storage: Arc<dyn KiroTokenStorage>,
) -> Result<(
    DeviceAuthorization,
    tokio::sync::oneshot::Receiver<Result<LoginResult>>,
)> {
    let region = DEFAULT_REGION.to_string();
    let oidc_endpoint = build_sso_oidc_endpoint(&region);
    let client = reqwest::Client::new();

    let registration = register_client(&client, &oidc_endpoint).await?;
    let device_auth = authorize_device(
        &client,
        &oidc_endpoint,
        &registration.client_id,
        &registration.client_secret,
        BUILDER_ID_START_URL,
    )
    .await?;

    let auth = DeviceAuthorization {
        verification_url: device_auth.verification_uri.clone(),
        verification_uri_complete: device_auth.verification_uri_complete.clone(),
        user_code: device_auth.user_code.clone(),
        interval_seconds: device_auth.interval,
        expires_in_seconds: device_auth.expires_in,
    };

    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let result = poll_for_tokens(
            &client,
            &oidc_endpoint,
            &registration.client_id,
            &registration.client_secret,
            &device_auth.device_code,
            device_auth.interval,
            device_auth.expires_in,
            &region,
            storage,
        )
        .await;
        let _ = tx.send(result);
    });

    Ok((auth, rx))
}

fn build_sso_oidc_endpoint(region: &str) -> String {
    format!("https://oidc.{region}.amazonaws.com")
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RegisterClientResponse {
    client_id: String,
    client_secret: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeviceAuthorizationResponse {
    verification_uri: String,
    verification_uri_complete: String,
    user_code: String,
    device_code: String,
    #[serde(default = "default_poll_interval_seconds")]
    interval: u64,
    #[serde(default = "default_device_code_expiry_seconds")]
    expires_in: u64,
}

fn default_poll_interval_seconds() -> u64 {
    5
}

fn default_device_code_expiry_seconds() -> u64 {
    600
}

#[derive(Debug)]
struct ParsedTokenPayload {
    access_token: Option<String>,
    refresh_token: Option<String>,
    expires_in: Option<i64>,
    error: Option<String>,
    error_description: Option<String>,
}

async fn register_client(
    client: &reqwest::Client,
    oidc_endpoint: &str,
) -> Result<RegisterClientResponse> {
    let response = client
        .post(format!("{oidc_endpoint}/client/register"))
        .header("Content-Type", "application/json")
        .header("User-Agent", KIRO_USER_AGENT)
        .json(&serde_json::json!({
            "clientName": "Kiro IDE",
            "clientType": "public",
            "scopes": KIRO_SCOPES,
            "grantTypes": [DEVICE_CODE_GRANT_TYPE, "refresh_token"],
        }))
        .send()
        .await
        .context("Failed to send Kiro client registration request")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!(
            "Kiro client registration failed with status {}: {}",
            status,
            body
        );
    }

    let parsed = response
        .json::<RegisterClientResponse>()
        .await
        .context("Failed to parse Kiro client registration response")?;

    if parsed.client_id.is_empty() || parsed.client_secret.is_empty() {
        bail!("Kiro client registration response missing client credentials");
    }

    Ok(parsed)
}

async fn authorize_device(
    client: &reqwest::Client,
    oidc_endpoint: &str,
    client_id: &str,
    client_secret: &str,
    start_url: &str,
) -> Result<DeviceAuthorizationResponse> {
    let response = client
        .post(format!("{oidc_endpoint}/device_authorization"))
        .header("Content-Type", "application/json")
        .header("User-Agent", KIRO_USER_AGENT)
        .json(&serde_json::json!({
            "clientId": client_id,
            "clientSecret": client_secret,
            "startUrl": start_url,
        }))
        .send()
        .await
        .context("Failed to send Kiro device authorization request")?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!(
            "Kiro device authorization failed with status {}: {}",
            status,
            body
        );
    }

    let parsed = response
        .json::<DeviceAuthorizationResponse>()
        .await
        .context("Failed to parse Kiro device authorization response")?;

    if parsed.device_code.is_empty()
        || parsed.user_code.is_empty()
        || parsed.verification_uri.is_empty()
        || parsed.verification_uri_complete.is_empty()
    {
        bail!("Kiro device authorization response missing required fields");
    }

    Ok(parsed)
}

async fn poll_for_tokens(
    client: &reqwest::Client,
    oidc_endpoint: &str,
    client_id: &str,
    client_secret: &str,
    device_code: &str,
    initial_interval_seconds: u64,
    expires_in_seconds: u64,
    region: &str,
    storage: Arc<dyn KiroTokenStorage>,
) -> Result<LoginResult> {
    let mut interval_seconds = initial_interval_seconds.max(1);
    let start = std::time::Instant::now();
    let expires_in = std::time::Duration::from_secs(expires_in_seconds.max(1));

    loop {
        if start.elapsed() >= expires_in {
            bail!("Kiro device login timed out. Please run `kiro-login` again.");
        }

        tokio::time::sleep(std::time::Duration::from_secs(interval_seconds)).await;

        let response = client
            .post(format!("{oidc_endpoint}/token"))
            .header("Content-Type", "application/json")
            .header("User-Agent", KIRO_USER_AGENT)
            .json(&serde_json::json!({
                "clientId": client_id,
                "clientSecret": client_secret,
                "deviceCode": device_code,
                "grantType": DEVICE_CODE_GRANT_TYPE,
            }))
            .send()
            .await
            .context("Failed to poll Kiro device token endpoint")?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let json_body = if body.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str::<Value>(&body)
                .with_context(|| format!("Invalid JSON from Kiro token endpoint: {body}"))?
        };

        let parsed = parse_token_payload(&json_body);

        if let Some(error) = parsed.error.as_deref() {
            match error {
                "authorization_pending" => continue,
                "slow_down" => {
                    interval_seconds = interval_seconds.saturating_add(5);
                    continue;
                }
                "expired_token" => {
                    bail!("Kiro device code expired before login completed.");
                }
                "access_denied" => {
                    bail!("Kiro authorization was denied.");
                }
                _ => {
                    let desc = parsed.error_description.unwrap_or_default();
                    bail!("Kiro token polling failed: {} {}", error, desc);
                }
            }
        }

        if let (Some(access_token), Some(refresh_token)) =
            (parsed.access_token, parsed.refresh_token)
        {
            let expires_at = parsed
                .expires_in
                .map(|seconds| Utc::now() + Duration::seconds(seconds));

            let auth_state = KiroAuthState {
                tokens: KiroTokens {
                    access_token,
                    refresh_token,
                    profile_arn: None,
                    expires_at,
                    client_id: Some(client_id.to_string()),
                    client_secret: Some(client_secret.to_string()),
                    region: Some(region.to_string()),
                    auth_method: Some(IDC_AUTH_METHOD.to_string()),
                },
                last_refresh: Utc::now(),
            };

            if let Err(e) = storage.save(&auth_state) {
                warn!("Could not save Kiro auth state (non-fatal): {e}");
            }

            return Ok(LoginResult { auth_state });
        }

        if !status.is_success() {
            bail!("Kiro token request failed with status {}: {}", status, body);
        }

        bail!("Kiro token polling response missing tokens.");
    }
}

fn parse_token_payload(payload: &Value) -> ParsedTokenPayload {
    ParsedTokenPayload {
        access_token: get_string(payload, &["access_token", "accessToken"]),
        refresh_token: get_string(payload, &["refresh_token", "refreshToken"]),
        expires_in: get_i64(payload, &["expires_in", "expiresIn"]),
        error: get_string(payload, &["error"]),
        error_description: get_string(payload, &["error_description", "errorDescription"]),
    }
}

fn get_string(payload: &Value, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| {
        payload
            .get(*key)
            .and_then(Value::as_str)
            .map(ToString::to_string)
    })
}

fn get_i64(payload: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter()
        .find_map(|key| payload.get(*key).and_then(Value::as_i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_sso_oidc_endpoint() {
        assert_eq!(
            build_sso_oidc_endpoint("us-east-1"),
            "https://oidc.us-east-1.amazonaws.com"
        );
    }

    #[test]
    fn test_load_auth_state_from_config_success() {
        let config = serde_json::json!({
            "kiro_tokens": {
                "access_token": "access-token",
                "refresh_token": "refresh-token",
                "expires_at": "2099-01-01T00:00:00Z",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "region": "us-west-2",
                "auth_method": "idc",
                "last_refresh": "2025-01-01T00:00:00Z"
            }
        });

        let state = load_auth_state_from_config(&config).expect("state should load");
        assert_eq!(state.tokens.access_token, "access-token");
        assert_eq!(state.tokens.refresh_token, "refresh-token");
        assert_eq!(state.tokens.client_id.as_deref(), Some("client-id"));
        assert_eq!(state.tokens.region.as_deref(), Some("us-west-2"));
    }

    #[test]
    fn test_load_auth_state_from_config_requires_access_token() {
        let missing_tokens = serde_json::json!({});
        assert!(load_auth_state_from_config(&missing_tokens).is_none());

        let empty_access_token = serde_json::json!({
            "kiro_tokens": {
                "access_token": "",
                "refresh_token": "refresh-token"
            }
        });
        assert!(load_auth_state_from_config(&empty_access_token).is_none());
    }

    #[test]
    fn test_needs_refresh_respects_expiry_grace_window() {
        let now = Utc::now();

        let stale = KiroAuthState {
            tokens: KiroTokens {
                access_token: "access".to_string(),
                refresh_token: "refresh".to_string(),
                profile_arn: None,
                expires_at: Some(now + Duration::minutes(EXPIRY_GRACE_MINUTES - 1)),
                client_id: None,
                client_secret: None,
                region: None,
                auth_method: None,
            },
            last_refresh: now,
        };
        assert!(stale.needs_refresh());

        let fresh = KiroAuthState {
            tokens: KiroTokens {
                access_token: "access".to_string(),
                refresh_token: "refresh".to_string(),
                profile_arn: None,
                expires_at: Some(now + Duration::minutes(EXPIRY_GRACE_MINUTES + 10)),
                client_id: None,
                client_secret: None,
                region: None,
                auth_method: None,
            },
            last_refresh: now,
        };
        assert!(!fresh.needs_refresh());
    }

    #[test]
    fn test_parse_token_payload_snake_case() {
        let payload = serde_json::json!({
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_in": 3600,
        });

        let parsed = parse_token_payload(&payload);
        assert_eq!(parsed.access_token.as_deref(), Some("access"));
        assert_eq!(parsed.refresh_token.as_deref(), Some("refresh"));
        assert_eq!(parsed.expires_in, Some(3600));
    }

    #[test]
    fn test_parse_token_payload_camel_case_with_error() {
        let payload = serde_json::json!({
            "error": "slow_down",
            "errorDescription": "back off",
            "accessToken": "ignored-access",
            "refreshToken": "ignored-refresh",
            "expiresIn": 1800,
        });

        let parsed = parse_token_payload(&payload);
        assert_eq!(parsed.error.as_deref(), Some("slow_down"));
        assert_eq!(parsed.error_description.as_deref(), Some("back off"));
        assert_eq!(parsed.access_token.as_deref(), Some("ignored-access"));
        assert_eq!(parsed.refresh_token.as_deref(), Some("ignored-refresh"));
        assert_eq!(parsed.expires_in, Some(1800));
    }
}
