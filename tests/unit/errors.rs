use llmr::errors::Error;

#[test]
fn test_user_facing_error_messages() {
    let missing_argument = Error::MissingArgument {
        arg: "--model".to_string(),
    };
    let timeout = Error::Timeout {
        message: "container health check timed out".to_string(),
    };
    let docker_error = Error::DockerError {
        message: "connection refused".to_string(),
    };

    assert_eq!(
        missing_argument.to_string(),
        "Missing required argument: --model"
    );
    assert_eq!(
        timeout.to_string(),
        "Timeout: container health check timed out"
    );
    assert_eq!(docker_error.to_string(), "Docker error: connection refused");
}

#[test]
fn test_io_error_conversion_preserves_source_message() {
    let err: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found").into();
    assert!(matches!(err, Error::IoError { .. }));
    assert!(err.to_string().contains("file not found"));
}

#[test]
fn test_toml_deserialization_error_conversion() {
    let parse_err = toml::from_str::<toml::Value>("not = [valid").unwrap_err();
    let err: Error = parse_err.into();

    assert!(matches!(err, Error::TomlDeError { .. }));
}

#[test]
fn test_toml_serialization_error_conversion() {
    let invalid_float = toml::Value::Float(f64::NAN);
    let err: Error = toml::to_string(&invalid_float).unwrap_err().into();

    assert!(matches!(err, Error::TomlSerError { .. }));
}
