//! Edge tests for Varlink protocol framing.

#[cfg(test)]
mod tests {
    use runtimed_daemon::varlink::{VarlinkCall, VarlinkReply};

    #[test]
    fn test_malformed_varlink_call() {
        let bad = b"{\"method\":}";
        let res: Result<VarlinkCall, _> = serde_json::from_slice(bad);
        assert!(res.is_err());
    }

    #[test]
    fn test_varlink_empty_error_reply() {
        let reply = VarlinkReply::err("RuntimePreempted", None);
        let bytes = reply.to_bytes();
        assert_eq!(*bytes.last().unwrap(), 0x00);
        let parsed: serde_json::Value =
            serde_json::from_slice(&bytes[..bytes.len() - 1]).unwrap();
        assert_eq!(parsed["error"], "RuntimePreempted");
    }
}
