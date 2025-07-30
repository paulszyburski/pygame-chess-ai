import zstandard as zstd

in_path  = "lichess_db.zst"
out_path = "lichess_db.pgn"

with open(in_path, "rb") as compressed, open(out_path, "wb") as dest:
    dctx = zstd.ZstdDecompressor()
    dctx.copy_stream(compressed, dest)
