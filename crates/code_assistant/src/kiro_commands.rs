//! CLI commands for Kiro Builder ID authentication.
//!
//! This command manages the device authorization flow and stores tokens in
//! `providers.json` under `config.kiro_tokens`.

use anyhow::Result;
use llm::kiro_auth::{self, KiroTokenStorage, ProvidersJsonTokenStorage};
use std::sync::Arc;

/// The default provider ID used for Kiro OAuth auth.
const PROVIDER_ID: &str = kiro_auth::DEFAULT_PROVIDER_ID;

/// Build the default providers.json storage backend.
fn default_storage() -> Arc<dyn KiroTokenStorage> {
    Arc::new(ProvidersJsonTokenStorage::new(
        PROVIDER_ID.to_string(),
        None,
    ))
}

/// Run the Kiro Builder ID device-code login flow.
pub async fn run_kiro_login() -> Result<()> {
    let storage = default_storage();

    let status = kiro_auth::get_auth_status(storage.as_ref());
    if status.authenticated {
        println!("Already logged in to Kiro.");
        if let Some(expires_at) = status.expires_at {
            println!("Token expiry: {}", expires_at.to_rfc3339());
        }
        println!("Run your provider logout flow first if you want to switch accounts.");
        return Ok(());
    }

    println!("Starting Kiro Builder ID login...");
    println!();

    let (device_auth, rx) = kiro_auth::start_login_flow(storage.clone()).await?;

    println!("Open this URL and complete login in your browser:");
    println!();
    println!("  {}", device_auth.verification_url);
    println!();
    println!("Then enter this code when prompted:");
    println!();
    println!("  {}", device_auth.user_code);
    println!();
    println!(
        "Waiting for authorization (up to {} seconds)...",
        device_auth.expires_in_seconds
    );

    if let Err(e) = open::that(&device_auth.verification_uri_complete) {
        eprintln!("Could not open browser automatically: {e}");
    }

    let result = rx.await??;

    println!();
    println!("Login successful!");
    if let Some(expires_at) = result.auth_state.tokens.expires_at {
        println!("  Expires At: {}", expires_at.to_rfc3339());
    }
    println!();
    println!("Tokens stored in providers.json under \"{}\".", PROVIDER_ID);
    println!("(The login flow creates this provider entry automatically if needed.)");

    Ok(())
}
