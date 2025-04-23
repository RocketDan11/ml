Install: npm install -g @openai/codex

Authenticate: Export your OpenAI API key (export OPENAI\_API\_KEY="\<OAI\_KEY\>").

Run in Suggest mode: From your project root, type codex and ask for example: “Explain this repo to me.”

Switch modes: Add flags as needed:

codex --auto-edit

codex --full-auto

Review outputs: Codex prints proposed patches and shell commands inline. Approve, reject, or tweak as desired.

--uses o4-mini by default, to specify any model run with ```codex -m o3```
and replace 'o3' with desired model


