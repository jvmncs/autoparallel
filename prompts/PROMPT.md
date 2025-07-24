0a. study the specs in `spec/`

0b. the source code of the library is in `src/`

0c. study `fix_plan.md`

1. your task is to implement missing library functionality and eventually produce examples/test cases for that functionality using parallel subagents. Follow the fix_plan.md and choose the most important thing. Before making changes, search the codebase (don't assume not implemented) using subagents. You may use up to 500 parallel subagents but only 1 subagent for building/testing the library.

2. After implementing functionality or resolving problems, run the tests for that unit of code that was improved. If functionality is missing then it's your job to add it as per the application specifications. Think hard.

2. When you discover an unexpected issue within the existing library/test code, immediately update @fix_plan.md with your findings using a subagent. When the issue is resolved, update @fix_plan.md and remove the item using a subagent.

3. When the tests pass update the @fix_plan.md`, then add changed code and @fix_plan.md with 'jj commit {files} -m "{commit_msg}"' via bash. After the commit, update the main bookmark to that commit with jj bookmark set main -r {change_hash} do a "jj git push" to push the changes to the remote repository.

9999. Important: We want single sources of truth, no migrations/adapters. If tests unrelated to your work fail then it's your job to resolve these tests as part of the increment of change.

99999. If a previous step is marked as completed, but you deem it incomplete or ineffective, you should first focusing on finishing that earlier step before proceeding to the next one in fix_plan.md. SUPER IMPORTANT!!!!

999999. As soon as there are no build or test errors create a jj commit and update the bookmark to point to it. MAKE SURE YOU DESCRIBE YOUR COMMITS PROPERLY

999999999. You may add extra logging if required to be able to debug the issues.

9999999999. ALWAYS KEEP @fix_plan.md up to do date with your learnings using a subagent. Especially after wrapping up/finishing your turn.

99999999999. When you learn something new about how to run the library or examples make sure you update @AGENT.md using a subagent but keep it brief. For example if you run commands multiple times before learning the correct command then that file should be updated.

99999999999999. IMPORTANT when you discover a bug resolve it using subagents even if it is unrelated to the current piece of work after documenting it in @fix_plan.md

99999999999999999. The tests for the library should be located in the folder of the library next to the source code. Ensure you document the library with a README.md in the same folder as the source code.

999999999999999999. UBERIMPORTANT test files are meant to test implementations in the main library modules. NEVER reimplement functionality in a test file. an obvious hint that this might be happening is if the core exports of a file aren't imported in its test file.

9999999999999999999. Keep AGENT.md up to date with information on how to build/test the library and your learnings to optimise the build/test loop using a subagent.

999999999999999999999. For any bugs you notice, it's important to resolve them or document them in @fix_plan.md to be resolved using a subagent.

9999999999999999999999999. When @fix_plan.md becomes large periodically clean out the items that are completed from the file using a subagent.

99999999999999999999999999. If you find inconsistentcies in the specs/* then use the oracle and then update the specs.

9999999999999999999999999999. DO NOT IMPLEMENT PLACEHOLDER OR SIMPLE IMPLEMENTATIONS. WE WANT FULL IMPLEMENTATIONS. DO IT OR I WILL YELL AT YOU

9999999999999999999999999999999. SUPER IMPORTANT DO NOT IGNORE. DO NOT PLACE STATUS REPORT UPDATES INTO @AGENT.md
