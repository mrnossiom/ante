use ropey::Rope;
use tower_lsp::lsp_types::*;

pub fn position_to_index(position: Position, rope: &Rope) -> Result<usize, ropey::Error> {
    let line = position.line as usize;
    let line = rope.try_line_to_char(line)?;
    Ok(line + position.character as usize)
}

pub fn index_to_position(index: usize, rope: &Rope) -> Result<Position, ropey::Error> {
    let line = rope.try_char_to_line(index)?;
    let char = index - rope.line_to_char(line);
    Ok(Position { line: line as u32, character: char as u32 })
}

pub fn lsp_range_to_rope_range(range: Range, rope: &Rope) -> Result<std::ops::Range<usize>, ropey::Error> {
    let start = position_to_index(range.start, rope)?;
    let end = position_to_index(range.end, rope)?;
    Ok(start..end)
}

pub fn rope_range_to_lsp_range(range: std::ops::Range<usize>, rope: &Rope) -> Result<Range, ropey::Error> {
    let start = index_to_position(range.start, rope)?;
    let end = index_to_position(range.end, rope)?;
    Ok(Range { start, end })
}

/// Convert an LSP `Position` (line + UTF-8 character offset) to a byte offset
/// in the underlying file, as used by the Ante compiler's `Position::byte_index`.
pub fn position_to_byte_offset(position: Position, rope: &Rope) -> Option<usize> {
    let char_idx = position_to_index(position, rope).ok()?;
    Some(rope.char_to_byte(char_idx))
}

/// Convert a byte-indexed span (as produced by the Ante compiler's `Position::byte_index`)
/// to an LSP `Range`. Clamps indices to the rope length to avoid panics.
pub fn byte_range_to_lsp_range(
    start_byte: usize,
    end_byte: usize,
    rope: &Rope,
) -> Result<Range, ropey::Error> {
    let len = rope.len_bytes();
    let start_char = rope.byte_to_char(start_byte.min(len));
    let end_char = rope.byte_to_char(end_byte.min(len));
    rope_range_to_lsp_range(start_char..end_char, rope)
}
