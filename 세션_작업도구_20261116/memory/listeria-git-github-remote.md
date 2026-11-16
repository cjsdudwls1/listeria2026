---
name: listeria-git-github-remote
description: "천영진 저장소 git 운영 — 2026-09-22 사용자 결정으로 로컬 전용 규칙 해제, GitHub 비공개 저장소 cjsdudwls1/listeria2026 을 origin 으로 복구·사용자 요청 시 push; worktree 미사용·data/collected 추적 제외·계정 파일 커밋 금지는 유지"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2ecac5d4-7adf-4d78-b00e-f06a53713835
  modified: 2026-09-22T09:02:22.784Z
---

2026-09-22 사용자 결정: 2026-09-17 의 "보안과제라 로컬 전용 git, GitHub 원격 삭제" 결정을 사용자가 취소했다("삭제. 그리고 올려."). 노트북 6 을 GitHub 에 올리라는 요청과 함께였다.

- 원격 `origin` = `https://github.com/cjsdudwls1/listeria2026.git` (비공개, gh CLI 계정 cjsdudwls1 로그인 상태). 2026-09-22 에 다시 추가했다.
- 원격 main 에는 2026-09-17 웹 업로드 커밋 2개(`리스테리아ML/_새판_컬럼사전.xlsx`, `리스테리아ML/질문_개정판.md`)가 있고 로컬 main 에는 없다(갈라짐). 2026-09-22 사용자가 "노트북만 올려"라고 해서 로컬 커밋·병합 없이 GitHub contents API(`gh api -X PUT`)로 노트북 6 파일만 원격 main 에 올렸다(커밋 a9482fc). 노트북 6 은 로컬에서 아직 미추적(untracked)이고 로컬 main 은 push 하지 않았다. 원격에 옛 `worktree-*` 브랜치 6개와 draft PR #1 이 남아 있다. 다음에 로컬 main 을 push 하려면 먼저 `origin/main` 을 병합해야 한다(force push 금지).
- 유지하는 것: worktree 미사용(`.claude/settings.json` `worktree.bgIsolation: none`), `리스테리아ML/data/collected/`·공유용 parquet 는 .gitignore 로 추적 제외(대용량·DB 원자료), `가이드.txt`(미추적)·`database계정.txt`(.gitignore) 는 절대 커밋하지 않는다.
- 노트북 6 등 출력이 든 노트북에는 사내 DB IP([IP 가림])·테이블명·실제 검사 데이터 집계가 들어 있다. 2026-09-22 에 사용자에게 이 점을 알렸고 사용자가 그대로 올리기로 했다.

**Why:** 사용자가 노트북을 GitHub 에 두고 싶어 하며, 보안 우려를 들은 뒤에도 올리기로 결정했다. 저장소는 비공개다.

**How to apply:** push 는 사용자가 요청할 때만. 커밋 전 `git status` 로 계정 정보 파일·사용자 편집 중 파일이 섞이지 않았는지 확인한다. 원격 main 과 로컬 main 이 갈라져 있으면 force push 대신 병합한다. 관련: [[listeria-share-dataset]], [[toxin-db-read-only-rule]].

**2026-10-22 PR #3 머지([1번 베이스라인 실험 다시])**: 사용자 지시 '전부 완료되면 깃 머지까지'에 따라 exp/1·2·3 코드·설정·결과 요약·표(2,122개 30MB)를 PR #3 → 머지(원격 main 017d9ca). 공유 체크아웃은 여러 세션이 같이 쓰므로 **브랜치를 바꾸지 않고** `GIT_INDEX_FILE=<임시> git read-tree origin/main` → `git add --pathspec-from-file` → `write-tree`/`commit-tree -p origin/main` → `git push origin <sha>:refs/heads/<새 브랜치>` → `gh pr create`·`gh pr merge --merge` → 로컬 main 은 공유 인덱스에 올린 변경이 0인지 확인 후 `git update-ref`(옛값 확인) + `git reset -q`(파일 안 건드림)로 맞춤. 뺀 것: parquet·npy/npz·tar·로그·cache/scratch/smoke·다른 세션 폴더(analysis/T기준_*). 커밋 전 비밀 문자열 검사(DB IP·/home/<계정>·키) — 서버에서 받은 config.json 의 홈 경로를 ~/cyj_listeria 로 가림.
