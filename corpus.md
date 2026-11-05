# Corpus and run data

Tasks are generated deterministically from seeds. No stored inputs or live-model captures are
included. Offline validation writes small `records.jsonl` files that can be regenerated with
`python3 -m lcb validate --out runs/validation`.

The `runs/` directory is ignored by git. Hermes runs include records, protocol transcripts,
workspaces, and copies of the home template. Those copies can contain API credentials in `.env`
or `config.yaml`; transcripts and generated skills may contain sensitive text as well.

Before publishing results, prepare a separate sanitized export. Remove credentials and private
content from every copied home, transcript, workspace and generated file. Review the complete
export, not only the records. Do not upload the raw run tree. Include the model, configuration
with secrets removed, exact Hermes revision, command, seed, condition, harness mode, file sizes,
and SHA-256 checksums with any shared results.
