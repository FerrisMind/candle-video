//! Feature-gated profiling helpers (tracing spans, optional Tracy zones).
//!
//! Default builds compile spans to no-ops. Enable with `--features profiling` or `tracy`.

#[cfg(feature = "profiling")]
#[macro_export]
macro_rules! profile_zone {
    ($name:expr) => {
        let _span = tracing::info_span!(target: "candle_video::profiling", "{}", $name).entered();
    };
    ($name:expr, $($field:tt)*) => {
        let _span = tracing::info_span!(target: "candle_video::profiling", $name, $($field)*).entered();
    };
}

#[cfg(not(feature = "profiling"))]
#[macro_export]
macro_rules! profile_zone {
    ($name:expr) => {};
    ($name:expr, $($field:tt)*) => {};
}

pub mod vram;

/// Initialize tracing subscriber for profiling builds.
pub fn init_tracing() {
    #[cfg(feature = "profiling")]
    {
        use tracing_subscriber::{EnvFilter, fmt};

        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("candle_video=info,candle_video::profiling=trace"));

        #[cfg(feature = "tracy")]
        {
            use tracing_subscriber::layer::SubscriberExt;
            let tracy = tracing_tracy::TracyLayer::default();
            let subscriber = tracing_subscriber::registry()
                .with(filter)
                .with(tracy)
                .with(fmt::layer().with_target(true));
            let _ = tracing::subscriber::set_global_default(subscriber);
        }

        #[cfg(not(feature = "tracy"))]
        {
            let _ = fmt().with_env_filter(filter).with_target(true).try_init();
        }
    }
}

/// Returns true when detailed profiling spans are compiled in.
pub fn profiling_enabled() -> bool {
    cfg!(feature = "profiling")
}
