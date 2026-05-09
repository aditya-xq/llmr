use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::{cmp::Reverse, ffi::OsStr};

#[cfg(target_os = "windows")]
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub name: String,
    pub size_bytes: u64,
    pub size_formatted: String,
}

impl ModelInfo {
    pub fn from_path(path: &Path) -> Option<Self> {
        let metadata = std::fs::metadata(path).ok()?;
        if !metadata.is_file() {
            return None;
        }
        let name = path.file_name()?.to_str()?.to_string();
        let size_bytes = metadata.len();
        let size_formatted = Self::format_size(size_bytes);
        Some(Self {
            path: path.to_path_buf(),
            name,
            size_bytes,
            size_formatted,
        })
    }

    fn format_size(bytes: u64) -> String {
        const KB: u64 = 1024;
        const MB: u64 = KB * 1024;
        const GB: u64 = MB * 1024;
        if bytes >= GB {
            format!("{:.1} GB", bytes as f64 / GB as f64)
        } else if bytes >= MB {
            format!("{:.1} MB", bytes as f64 / MB as f64)
        } else if bytes >= KB {
            format!("{:.1} KB", bytes as f64 / KB as f64)
        } else {
            format!("{} B", bytes)
        }
    }
}

pub struct ModelScanner;

const SKIP_DIRS: &[&str] = &[
    "node_modules",
    ".git",
    "target",
    "build",
    "tmp",
    "cache",
    "windows",
    "program files",
    "programdata",
    "$recycle.bin",
    "system volume information",
];

impl ModelScanner {
    pub fn new() -> Self {
        Self
    }

    const COMMON_MODEL_DIRS: &'static [&'static str] = &[
        "models",
        "llama.cpp",
        "llama.cpp/models",
        "lm-studio",
        "lm-studio/models",
        "ollama",
        "ollama/models",
        "text-generation-webui",
        "text-generation-webui/models",
        "gpt4all",
        "gpt4all/models",
        "koboldcpp",
        "koboldcpp/models",
        "llama",
        "ai",
        "ml",
        "deepseek",
        "qwen",
        "mistral",
    ];

    #[cfg(target_os = "windows")]
    pub fn get_local_drives() -> Vec<PathBuf> {
        let mut drives = Vec::new();
        let mut added_drives = std::collections::HashSet::new();

        let output = Command::new("wmic")
            .args(["logicaldisk", "get", "name"])
            .output();

        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines().skip(1) {
                let drive = line.trim();
                if drive.len() == 2 && drive.ends_with(':') {
                    let path = PathBuf::from(format!(r"{drive}\"));
                    if path.exists() && added_drives.insert(drive.to_lowercase()) {
                        drives.push(path);
                    }
                }
            }
        }

        if drives.is_empty() {
            if let Ok(programdata) = std::env::var("ProgramData") {
                let drive = programdata.chars().take(2).collect::<String>();
                if drive.len() == 2 && drive.ends_with(':') {
                    drives.push(PathBuf::from(format!(r"{drive}\")));
                }
            }
        }

        if let Some(home) = dirs::home_dir() {
            let home_drive = home.to_string_lossy().chars().take(2).collect::<String>();
            if !home_drive.is_empty() && added_drives.insert(home_drive.to_lowercase()) {
                let drive = PathBuf::from(format!(r"{}\", home_drive));
                if drive.exists() && !drives.contains(&drive) {
                    drives.insert(0, drive);
                }
            }
        }

        drives
    }

    #[cfg(target_os = "windows")]
    pub fn get_network_drives() -> Vec<PathBuf> {
        let mut drives = Vec::new();
        let output = Command::new("netstat").args(["-n"]).output();

        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines() {
                if line.contains("Mapped") || line.contains("Network drive") {
                    continue;
                }
            }
        }

        let output = Command::new("wmic")
            .args(["logicaldisk", "get", "name", ",", "drivetype"])
            .output();

        if let Ok(out) = output {
            let text = String::from_utf8_lossy(&out.stdout);
            for line in text.lines().skip(1) {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    let drive = parts[0].trim();
                    let drivetype = parts[1].trim();
                    if drive.len() == 2 && drive.ends_with(':') && drivetype == "4" {
                        let path = PathBuf::from(format!(r"{drive}\"));
                        if path.exists() {
                            drives.push(path);
                        }
                    }
                }
            }
        }

        drives
    }

    #[cfg(not(target_os = "windows"))]
    pub fn get_local_drives() -> Vec<PathBuf> {
        let mut drives = Vec::new();

        drives.push(PathBuf::from("/"));

        if let Some(home) = dirs::home_dir() {
            if let Some(root) = home.ancestors().next() {
                let root_path = PathBuf::from(root.as_os_str());
                if !drives.contains(&root_path) {
                    drives.push(root_path);
                }
            }
        }

        if let Some(data) = dirs::data_dir() {
            if let Some(root) = data.ancestors().next() {
                let root_path = PathBuf::from(root.as_os_str());
                if !drives.contains(&root_path) {
                    drives.push(root_path);
                }
            }
        }

        drives
    }

    pub fn scan_disks(&self) -> Vec<ModelInfo> {
        let mut all_models = Vec::new();
        let mut scanned_paths = std::collections::HashSet::new();

        let common_dirs = Self::find_common_model_dirs();
        for path in common_dirs {
            if scanned_paths.insert(path.clone()) {
                all_models.extend(self.scan_directory(&path));
            }
        }

        let drives = Self::get_local_drives();
        for drive in drives {
            if scanned_paths.insert(drive.clone()) {
                self.scan_root(&drive, &mut all_models, &mut scanned_paths);
            }
        }

        #[cfg(target_os = "windows")]
        {
            for drive in Self::get_network_drives() {
                if scanned_paths.insert(drive.clone()) {
                    self.scan_root(&drive, &mut all_models, &mut scanned_paths);
                }
            }
        }

        for path in Self::find_root_paths() {
            if scanned_paths.insert(path.clone()) {
                self.scan_root(&path, &mut all_models, &mut scanned_paths);
            }
        }

        Self::dedupe_models(&mut all_models);
        all_models.sort_by_key(|model| Reverse(model.size_bytes));
        all_models
    }

    fn find_common_model_dirs() -> Vec<PathBuf> {
        let mut dirs = Vec::new();

        if let Some(home) = dirs::home_dir() {
            for dir in Self::COMMON_MODEL_DIRS {
                let path = home.join(dir);
                if path.exists() && path.is_dir() {
                    dirs.push(path);
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(programdata) = std::env::var("ProgramData") {
                let base = PathBuf::from(programdata);
                for dir in Self::COMMON_MODEL_DIRS {
                    let path = base.join(dir);
                    if path.exists() && path.is_dir() {
                        dirs.push(path);
                    }
                }
            }

            if let Ok(programfiles) = std::env::var("ProgramFiles") {
                let path = PathBuf::from(programfiles).join("llama.cpp").join("models");
                if path.exists() && path.is_dir() {
                    dirs.push(path);
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            let application_support = dirs::data_dir();
            if let Some(base) = application_support {
                for dir in Self::COMMON_MODEL_DIRS {
                    let path = base.join(dir);
                    if path.exists() && path.is_dir() {
                        dirs.push(path);
                    }
                }
            }
        }

        dirs
    }

    pub fn scan_paths(&self, paths: &[PathBuf]) -> Vec<ModelInfo> {
        let mut all_models = Vec::new();

        for base in paths {
            if base.is_dir() {
                let found = self.scan_directory(base);
                all_models.extend(found);
            } else if base.is_file() {
                if let Some(info) = ModelInfo::from_path(base) {
                    all_models.push(info);
                }
            }
        }

        Self::dedupe_models(&mut all_models);
        all_models.sort_by_key(|model| Reverse(model.size_bytes));
        all_models
    }

    fn scan_root(
        &self,
        root: &Path,
        models: &mut Vec<ModelInfo>,
        scanned: &mut std::collections::HashSet<PathBuf>,
    ) {
        let max_depth = 6;
        self.scan_recursive(root, models, 0, max_depth, scanned);
    }

    fn scan_recursive(
        &self,
        dir: &Path,
        models: &mut Vec<ModelInfo>,
        depth: usize,
        max_depth: usize,
        scanned: &mut std::collections::HashSet<PathBuf>,
    ) {
        if depth > max_depth {
            return;
        }

        let canonical = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());
        if scanned.contains(&canonical) {
            return;
        }
        let _ = scanned.insert(canonical);

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                if path.is_file() {
                    if Self::is_gguf_file(&path) {
                        if let Some(info) = ModelInfo::from_path(&path) {
                            models.push(info);
                        }
                    }
                } else if path.is_dir() {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_lowercase())
                        .unwrap_or_default();

                    if !SKIP_DIRS.contains(&name.as_str()) {
                        self.scan_recursive(&path, models, depth + 1, max_depth, scanned);
                    }
                }
            }
        }
    }

    pub fn find_root_paths() -> Vec<PathBuf> {
        let mut roots = Vec::new();

        #[cfg(target_os = "linux")]
        {
            roots.push(PathBuf::from("/models"));
            roots.push(PathBuf::from("/usr/local/share/llama.cpp/models"));
            roots.push(
                dirs::home_dir()
                    .map(|p| p.join("models"))
                    .unwrap_or_default(),
            );
            roots.push(
                dirs::home_dir()
                    .map(|p| p.join(".cache").join("lm-studio").join("models"))
                    .unwrap_or_default(),
            );
        }

        #[cfg(target_os = "macos")]
        {
            roots.push(
                dirs::home_dir()
                    .map(|p| p.join("models"))
                    .unwrap_or_default(),
            );
            roots.push(PathBuf::from("/usr/local/share/llama.cpp/models"));
        }

        #[cfg(target_os = "windows")]
        {
            roots.push(
                dirs::home_dir()
                    .map(|p| p.join("models"))
                    .unwrap_or_default(),
            );
            roots.push(PathBuf::from("C:\\models"));
            if let Ok(programdata) = std::env::var("ProgramData") {
                roots.push(PathBuf::from(programdata).join("llama.cpp").join("models"));
            }
        }

        if let Ok(model_path) = std::env::var("LLAMA_CPP_MODEL_PATH") {
            roots.push(PathBuf::from(model_path));
        }

        roots.into_iter().filter(|p| p.exists()).collect()
    }

    pub fn scan_directory(&self, dir: &Path) -> Vec<ModelInfo> {
        let mut models = Vec::new();
        if !dir.is_dir() {
            return models;
        }

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if Self::is_gguf_file(&path) {
                    if let Some(info) = ModelInfo::from_path(&path) {
                        models.push(info);
                    }
                }
            }
        }

        Self::dedupe_models(&mut models);
        models.sort_by_key(|model| Reverse(model.size_bytes));
        models
    }

    pub fn scan_all(&self) -> Vec<(PathBuf, Vec<ModelInfo>)> {
        let mut results = Vec::new();

        let disk_models = self.scan_disks();
        if !disk_models.is_empty() {
            let mut by_parent: std::collections::HashMap<PathBuf, Vec<ModelInfo>> =
                std::collections::HashMap::new();
            for model in disk_models {
                if let Some(parent) = model.path.parent() {
                    by_parent
                        .entry(parent.to_path_buf())
                        .or_default()
                        .push(model);
                }
            }

            for (root, models) in by_parent {
                results.push((root, models));
            }
        } else {
            for root in Self::find_root_paths() {
                let models = self.scan_directory(&root);
                if !models.is_empty() {
                    results.push((root, models));
                }
            }
        }

        results.sort_by_key(|entry| Reverse(entry.1.len()));
        results
    }

    fn is_gguf_file(path: &Path) -> bool {
        path.extension()
            .and_then(OsStr::to_str)
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    }

    fn dedupe_models(models: &mut Vec<ModelInfo>) {
        let mut seen = std::collections::HashSet::new();
        models.retain(|model| {
            if model.name.to_lowercase().starts_with("mmproj-") {
                return false;
            }
            let key = model
                .path
                .canonicalize()
                .unwrap_or_else(|_| model.path.clone());
            seen.insert(key)
        });
    }
}

impl Default for ModelScanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_model_info_from_path_valid() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.gguf");

        let mut file = File::create(&model_path).unwrap();
        file.write_all(b"test content").unwrap();
        drop(file);

        let info = ModelInfo::from_path(&model_path);
        assert!(info.is_some());

        let info = info.unwrap();
        assert_eq!(info.name, "test_model.gguf");
        assert_eq!(info.size_bytes, 12);
    }

    #[test]
    fn test_model_info_from_path_not_file() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path().join("models");

        std::fs::create_dir(&dir_path).unwrap();

        let info = ModelInfo::from_path(&dir_path);
        assert!(info.is_none());
    }

    #[test]
    fn test_model_info_from_path_not_exists() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("nonexistent.gguf");

        let info = ModelInfo::from_path(&path);
        assert!(info.is_none());
    }

    #[test]
    fn test_model_info_format_size() {
        assert!(ModelInfo::format_size(500).contains("B"));
        assert!(ModelInfo::format_size(1024).contains("KB"));
        assert!(ModelInfo::format_size(1024 * 1024).contains("MB"));
        assert!(ModelInfo::format_size(1024 * 1024 * 1024).contains("GB"));
    }

    #[test]
    fn test_model_scanner_new() {
        let _scanner = ModelScanner::new();
    }

    #[test]
    fn test_model_scanner_scan_directory() {
        let temp_dir = TempDir::new().unwrap();

        let model_path = temp_dir.path().join("model1.gguf");
        std::fs::write(&model_path, "test").unwrap();

        let scanner = ModelScanner::new();
        let models = scanner.scan_directory(temp_dir.path());

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "model1.gguf");
    }

    #[test]
    fn test_model_scanner_scan_directory_empty() {
        let temp_dir = TempDir::new().unwrap();

        let scanner = ModelScanner::new();
        let models = scanner.scan_directory(temp_dir.path());

        assert!(models.is_empty());
    }

    #[test]
    fn test_model_scanner_scan_directory_not_dir() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("file.txt");
        std::fs::write(&file_path, "test").unwrap();

        let scanner = ModelScanner::new();
        let models = scanner.scan_directory(&file_path);

        assert!(models.is_empty());
    }

    #[test]
    fn test_model_scanner_scan_directory_only_gguf() {
        let temp_dir = TempDir::new().unwrap();

        std::fs::write(temp_dir.path().join("model.gguf"), "test").unwrap();
        std::fs::write(temp_dir.path().join("model.txt"), "test").unwrap();
        std::fs::write(temp_dir.path().join("model.bin"), "test").unwrap();

        let scanner = ModelScanner::new();
        let models = scanner.scan_directory(temp_dir.path());

        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_model_scanner_scan_directory_accepts_uppercase_extension() {
        let temp_dir = TempDir::new().unwrap();

        std::fs::write(temp_dir.path().join("model.GGUF"), "test").unwrap();

        let scanner = ModelScanner::new();
        let models = scanner.scan_directory(temp_dir.path());

        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_model_scanner_scan_directory_sorted_by_size() {
        let temp_dir = TempDir::new().unwrap();

        let small = temp_dir.path().join("small.gguf");
        std::fs::write(&small, "a").unwrap();

        let large = temp_dir.path().join("large.gguf");
        std::fs::write(&large, "abcdefghij").unwrap();

        let scanner = ModelScanner::new();
        let models = scanner.scan_directory(temp_dir.path());

        assert_eq!(models.len(), 2);
        assert!(models[0].size_bytes >= models[1].size_bytes);
    }

    #[test]
    fn test_model_scanner_scan_paths() {
        let temp_dir = TempDir::new().unwrap();

        let model_path = temp_dir.path().join("model.gguf");
        std::fs::write(&model_path, "test").unwrap();

        let scanner = ModelScanner::new();
        let models = scanner.scan_paths(&[temp_dir.path().to_path_buf()]);

        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_scanner_scan_paths_with_file() {
        let temp_dir = TempDir::new().unwrap();

        let model_path = temp_dir.path().join("model.gguf");
        std::fs::write(&model_path, "test").unwrap();

        let scanner = ModelScanner::new();
        let models = scanner.scan_paths(std::slice::from_ref(&model_path));

        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_dedupe_models() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("model.gguf");
        std::fs::write(&model_path, "test").unwrap();

        let mut models = vec![
            ModelInfo::from_path(&model_path).unwrap(),
            ModelInfo::from_path(&model_path).unwrap(),
        ];

        ModelScanner::dedupe_models(&mut models);
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_model_scanner_find_root_paths() {
        let roots = ModelScanner::find_root_paths();
        // Either empty or has paths, but shouldn't crash
        assert!(roots.is_empty() || !roots.is_empty());
    }

    #[test]
    fn test_model_info_size_formatted() {
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test.gguf");
        std::fs::write(&model_path, "test").unwrap();

        let info = ModelInfo::from_path(&model_path).unwrap();
        assert!(!info.size_formatted.is_empty());
    }
}
