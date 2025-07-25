study specs/* to learn about the library specifications and fix_plan.md to understand plan so far.

The source code of the library is in src/*. Study them.

First task is to study @fix_plan.md (it may be incorrect) and to use up to 500 parallel subagents to study existing source code in src/ and compare it against the library specifications. Consult the oracle to look for internal contradictions in the source code/existing specs. From that research, create a new spec that simplifies the existing implementation and library as much as possible. Keep it DRY. Update the existing specs with their simplified forms, or delete them if they're no longer relevant.

Second task, create a @fix_plan.md, which is a bullet point list sorted in priority of the items which have yet to be implemeneted. It's okay to throw away the current one and start from scratch, but it MUST be based on the newly simplified specs in spec/. Think extra hard and use the oracle to plan the updated @fix_plan.md. Study @fix_plan.md to determine starting point for research and keep it up to date with items considered complete/incomplete using subagents.
