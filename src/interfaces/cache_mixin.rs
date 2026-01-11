pub trait CacheMixin {
    fn enable_caching(&mut self) {}

    fn disable_caching(&mut self) {}

    fn clear_cache(&mut self) {}
}

pub fn apply_cache_mixin<T: CacheMixin>(model: &mut T) {
    if std::env::var("CANDLE_VIDEO_CACHE_ENABLE").is_ok() {
        model.enable_caching();
    }
    if std::env::var("CANDLE_VIDEO_CACHE_CLEAR").is_ok() {
        model.clear_cache();
    }
}
