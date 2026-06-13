//! Prompt cleaning matching Diffusers `WanPipeline.prompt_clean`.

use regex::Regex;
use std::sync::LazyLock;

static WHITESPACE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+").expect("valid whitespace regex"));

/// Basic HTML entity unescape (double-pass like Diffusers; ftfy optional at call site).
pub fn basic_clean(text: &str) -> String {
    let once = unescape_html_entities(text);
    unescape_html_entities(&once).trim().to_string()
}

fn unescape_html_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

/// Collapse whitespace runs to a single space.
pub fn whitespace_clean(text: &str) -> String {
    WHITESPACE_RE.replace_all(text.trim(), " ").to_string()
}

/// Full Wan prompt normalization pipeline.
pub fn prompt_clean(text: &str) -> String {
    whitespace_clean(&basic_clean(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_clean_collapses_whitespace() {
        assert_eq!(prompt_clean("  hello   world  "), "hello world");
    }
}
