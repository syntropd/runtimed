//! Continuous streaming log journal with bounded memory and sink preservation.

use std::collections::VecDeque;

/// Individual entry in the streaming journal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalEntry {
    /// Monotonically increasing entry sequence identifier.
    pub id: u64,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Text content of this log or streaming event.
    pub text: String,
    /// Whether this entry is pinned as an immutable attention sink.
    pub is_sink: bool,
}

/// Continuous log journal maintaining pinned sink entries and a bounded rolling window.
pub struct StreamJournal {
    sink_entries: Vec<JournalEntry>,
    window_entries: VecDeque<JournalEntry>,
    max_window_entries: usize,
    max_window_bytes: usize,
    current_window_bytes: usize,
    next_id: u64,
}

impl StreamJournal {
    /// Create a new journal with explicit limits on rolling window entries and byte footprint.
    pub fn new(max_window_entries: usize, max_window_bytes: usize) -> Self {
        Self {
            sink_entries: Vec::new(),
            window_entries: VecDeque::new(),
            max_window_entries: max_window_entries.max(1),
            max_window_bytes: max_window_bytes.max(1),
            current_window_bytes: 0,
            next_id: 1,
        }
    }

    /// Append a new log line or event to the journal.
    pub fn append(&mut self, text: impl Into<String>, is_sink: bool, timestamp_ms: u64) -> u64 {
        let text = text.into();
        let bytes = text.len();
        let id = self.next_id;
        self.next_id += 1;

        let entry = JournalEntry {
            id,
            timestamp_ms,
            text,
            is_sink,
        };

        if is_sink {
            self.sink_entries.push(entry);
            return id;
        }

        self.window_entries.push_back(entry);
        self.current_window_bytes += bytes;

        // Evict oldest entries to prevent out-of-memory under unbounded continuous streams
        while self.window_entries.len() > self.max_window_entries
            || (self.current_window_bytes > self.max_window_bytes && self.window_entries.len() > 1)
        {
            if let Some(evicted) = self.window_entries.pop_front() {
                self.current_window_bytes = self
                    .current_window_bytes
                    .saturating_sub(evicted.text.len());
            }
        }

        id
    }

    /// Retrieve all entries with id strictly greater than `cursor`.
    pub fn entries_since(&self, cursor: u64) -> Vec<&JournalEntry> {
        let mut out = Vec::new();
        for entry in &self.sink_entries {
            if entry.id > cursor {
                out.push(entry);
            }
        }
        for entry in &self.window_entries {
            if entry.id > cursor {
                out.push(entry);
            }
        }
        out
    }

    /// Render composite prompt context: pinned sink header followed by rolling window.
    pub fn render_context(&self) -> String {
        let mut out = String::new();
        for sink in &self.sink_entries {
            out.push_str(&sink.text);
            out.push('\n');
        }
        if !self.sink_entries.is_empty() && !self.window_entries.is_empty() {
            out.push_str("--- [Recent Logs] ---\n");
        }
        for win in &self.window_entries {
            out.push_str(&win.text);
            out.push('\n');
        }
        out
    }

    pub fn sink_count(&self) -> usize {
        self.sink_entries.len()
    }

    pub fn window_count(&self) -> usize {
        self.window_entries.len()
    }

    pub fn current_window_bytes(&self) -> usize {
        self.current_window_bytes
    }

    pub fn clear_window(&mut self) {
        self.window_entries.clear();
        self.current_window_bytes = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_window_eviction() {
        let mut journal = StreamJournal::new(3, 10_000);
        let _id1 = journal.append("line 1", false, 100);
        let id2 = journal.append("line 2", false, 200);
        let id3 = journal.append("line 3", false, 300);
        assert_eq!(journal.window_count(), 3);

        let id4 = journal.append("line 4", false, 400);
        assert_eq!(journal.window_count(), 3);

        let entries = journal.entries_since(0);
        let ids: Vec<u64> = entries.iter().map(|e| e.id).collect();
        // Oldest entry (line 1) evicted
        assert_eq!(ids, vec![id2, id3, id4]);
    }

    #[test]
    fn test_sink_entries_preserved_across_evictions() {
        let mut journal = StreamJournal::new(2, 10_000);
        let s1 = journal.append("system prompt", true, 50);
        let _w1 = journal.append("log 1", false, 100);
        let _w2 = journal.append("log 2", false, 200);
        let _w3 = journal.append("log 3", false, 300);

        assert_eq!(journal.sink_count(), 1);
        assert_eq!(journal.window_count(), 2);

        let entries = journal.entries_since(0);
        assert_eq!(entries[0].id, s1);
        assert!(entries[0].is_sink);
    }

    #[test]
    fn test_byte_limit_eviction() {
        let mut journal = StreamJournal::new(100, 20);
        // Each entry is 10 bytes ("0123456789")
        journal.append("0123456789", false, 1);
        journal.append("abcdefghij", false, 2);
        assert_eq!(journal.window_count(), 2);
        // Adding third entry pushes byte size to 30 > 20, evicts oldest
        journal.append("klmnopqrst", false, 3);
        assert!(journal.current_window_bytes() <= 20);
        assert_eq!(journal.window_count(), 2);
    }

    #[test]
    fn test_render_context_formatting() {
        let mut journal = StreamJournal::new(5, 10_000);
        journal.append("SYSTEM: Monitor daemon active", true, 10);
        journal.append("EVENT: Node 1 ping ok", false, 20);
        let rendered = journal.render_context();
        assert!(rendered.contains("SYSTEM: Monitor daemon active"));
        assert!(rendered.contains("--- [Recent Logs] ---"));
        assert!(rendered.contains("EVENT: Node 1 ping ok"));
    }
}
