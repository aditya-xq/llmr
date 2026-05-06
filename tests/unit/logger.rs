use llmr::utils::logger::{gpu_style, Style};

#[test]
fn test_style_default() {
    let style = Style::plain();
    assert_eq!(style.success("ok"), "ok");
}

#[test]
fn test_style_new() {
    let style = Style::new();
    assert!(style.title("hello").contains("hello"));
}

#[test]
fn test_gpu_style_nvidia() {
    let result = gpu_style(&Style::plain(), "NVIDIA RTX 3080");
    assert!(result.contains("NVIDIA RTX 3080"));
}

#[test]
fn test_gpu_style_amd() {
    let result = gpu_style(&Style::plain(), "AMD Radeon RX 6800");
    assert!(result.contains("AMD Radeon RX 6800"));
}

#[test]
fn test_gpu_style_intel() {
    let result = gpu_style(&Style::plain(), "Intel Iris Xe");
    assert!(result.contains("Intel Iris Xe"));
}

#[test]
fn test_gpu_style_unknown() {
    let result = gpu_style(&Style::plain(), "Vulkan GPU");
    assert!(result.contains("Vulkan GPU"));
}

#[test]
fn test_plain_style_has_no_ansi() {
    let style = Style::plain();
    let output = style.warning("Warning");
    assert_eq!(output, "Warning");
    assert!(!output.contains("\x1b["));
}
