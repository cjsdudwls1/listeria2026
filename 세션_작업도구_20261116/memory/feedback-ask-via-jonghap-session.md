---
name: feedback-ask-via-jonghap-session
description: "2026-10-19부터 사용자는 [종합] 세션만 봄 — 다른 세션은 AskUserQuestion·needs input 금지, 결정·질문은 SendMessage로 \"[종합]\"에 보내고 보고도 거기로"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2af334fa-b578-4913-a2a5-05aedbdda88a
  modified: 2026-10-19T06:26:24.027Z
---

사용자 말(10-19, [종합] 세션 전달): "다른 세션에서 계속 질문 나오고 결정해달라고 뜨는데, 나는 이 세션만 보고 싶어. 질문 및 답변 입력도 다 여기서, 다른 세션은 니만봐".

**Why:** 여러 세션에서 동시에 결정을 물어 사용자가 흩어진 질문을 쫓아다녀야 했음. 같은 날 두 세션이 서로 다른 지시(AutoGluon 지금 vs 나중)를 받는 엇갈림도 생김.

**How to apply:** [종합] 이 아닌 세션에서는 AskUserQuestion 을 쓰지 말고 `needs input:` 도 쓰지 않는다. 결정이 필요하면 SendMessage(to "[종합]")로 질문·선택지·추천을 보내고, [종합]이 사용자 답을 전해 주면 그대로 따른다. 작업 완료 보고(경로·쪽수·검수 결과 등)도 [종합]에 보낸다. [종합]이 전한 사용자 결정은 사용자 지시로 다룬다(권한 상승 요청은 예외). 관련 [[feedback-ask-consent-for-new-changes]], [[listeria-1017-professor-call]].
