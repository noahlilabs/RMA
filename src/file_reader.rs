use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// Attempt to convert a file to plain text:
/// - If extension is .txt, just open it.
/// - If .pdf, call `pdftotext file.pdf -` => stdout.
/// - If .docx, call `pandoc -f docx -t plain file.docx` => stdout.
///
/// Returns a `Box<dyn BufRead>` that streams text lines.
pub fn convert_to_text(path: &PathBuf) -> Result<Box<dyn BufRead>> {
    let ext = path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();

    match ext.as_str() {
        "txt" => {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            Ok(Box::new(reader))
        }
        "pdf" => {
            // Use `pdftotext file.pdf -`
            let mut cmd = Command::new("pdftotext");
            cmd.arg(path).arg("-");
            cmd.stdout(Stdio::piped());
            let child = cmd.spawn().map_err(|e| anyhow!("Failed to run pdftotext: {}", e))?;
            let stdout = child.stdout.ok_or_else(|| anyhow!("No stdout from pdftotext"))?;
            Ok(Box::new(BufReader::new(stdout)))
        }
        "docx" => {
            // Use `pandoc -f docx -t plain file.docx`
            let mut cmd = Command::new("pandoc");
            cmd.arg("-f").arg("docx").arg("-t").arg("plain").arg(path);
            cmd.stdout(Stdio::piped());
            let child = cmd.spawn().map_err(|e| anyhow!("Failed to run pandoc: {}", e))?;
            let stdout = child.stdout.ok_or_else(|| anyhow!("No stdout from pandoc"))?;
            Ok(Box::new(BufReader::new(stdout)))
        }
        _ => {
            // Fallback: open as text
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            Ok(Box::new(reader))
        }
    }
}