// [메모] 이미지와 PDF 업로드 이후 
서버 오류가 발생했습니다
xhr.js:198  POST http://localhost:3000/api/ai/interview/resume/upload 500 (Internal Server Error){
    "status": "error",
    "message": "서버 오류가 발생했습니다",
    "error": "MaxUploadSizeExceededException: Maximum upload size exceeded",
    "timestamp": "2026-03-17T13:03:18.7492126",
    "path": "/api/ai/interview/resume/upload"
}

PDF 업로드
xhr.js:198  POST http://localhost:3000/api/ai/interview/resume/upload net::ERR_CONNECTION_ABORTED
dispatchXhrRequest @ xhr.js:198
xhr @ xhr.js:15
dispatchRequest @ dispatchRequest.js:51
Promise.then
_request @ Axios.js:172
request @ Axios.js:41
httpMethod @ Axios.js:233
wrap @ bind.js:12
uploadFile @ AIInterviewIntro.tsx:54
handlePdfFile @ AIInterviewIntro.tsx:80
executeDispatch @ react-dom-client.development.js:20543
runWithFiberInDEV @ react-dom-client.development.js:986
processDispatchQueue @ react-dom-client.development.js:20593
(익명) @ react-dom-client.development.js:21164
batchedUpdates$1 @ react-dom-client.development.js:3377
dispatchEventForPluginEventSystem @ react-dom-client.development.js:20747
dispatchEvent @ react-dom-client.development.js:25693
dispatchDiscreteEvent @ react-dom-client.development.js:25661
<input>
exports.jsxDEV @ react-jsx-dev-runtime.development.js:342
AIInterviewIntro @ AIInterviewIntro.tsx:188
react_stack_bottom_frame @ react-dom-client.development.js:28038
renderWithHooksAgain @ react-dom-client.development.js:8084
renderWithHooks @ react-dom-client.development.js:7996
updateFunctionComponent @ react-dom-client.development.js:10501
beginWork @ react-dom-client.development.js:12136
runWithFiberInDEV @ react-dom-client.development.js:986
performUnitOfWork @ react-dom-client.development.js:18997
workLoopSync @ react-dom-client.development.js:18825
renderRootSync @ react-dom-client.development.js:18806
performWorkOnRoot @ react-dom-client.development.js:17835
performSyncWorkOnRoot @ react-dom-client.development.js:20399
flushSyncWorkAcrossRoots_impl @ react-dom-client.development.js:20241
processRootScheduleInMicrotask @ react-dom-client.development.js:20280
(익명) @ react-dom-client.development.js:20418
<AIInterviewIntro>
exports.jsxDEV @ react-jsx-dev-runtime.development.js:342
renderIntro @ AIInterviewComponent.tsx:59
AIInterviewComponent @ AIInterviewComponent.tsx:113
react_stack_bottom_frame @ react-dom-client.development.js:28038
renderWithHooksAgain @ react-dom-client.development.js:8084
renderWithHooks @ react-dom-client.development.js:7996
updateFunctionComponent @ react-dom-client.development.js:10501
beginWork @ react-dom-client.development.js:12136
runWithFiberInDEV @ react-dom-client.development.js:986
performUnitOfWork @ react-dom-client.development.js:18997
workLoopSync @ react-dom-client.development.js:18825
renderRootSync @ react-dom-client.development.js:18806
performWorkOnRoot @ react-dom-client.development.js:17835
performWorkOnRootViaSchedulerTask @ react-dom-client.development.js:20384
performWorkUntilDeadline @ scheduler.development.js:45
<AIInterviewComponent>
exports.jsxDEV @ react-jsx-dev-runtime.development.js:342
WrappedComponent @ withRenderTrack.tsx:15
react_stack_bottom_frame @ react-dom-client.development.js:28038
renderWithHooksAgain @ react-dom-client.development.js:8084
renderWithHooks @ react-dom-client.development.js:7996
updateFunctionComponent @ react-dom-client.development.js:10501
beginWork @ react-dom-client.development.js:12136
runWithFiberInDEV @ react-dom-client.development.js:986
performUnitOfWork @ react-dom-client.development.js:18997
workLoopSync @ react-dom-client.development.js:18825
renderRootSync @ react-dom-client.development.js:18806
performWorkOnRoot @ react-dom-client.development.js:17835
performWorkOnRootViaSchedulerTask @ react-dom-client.development.js:20384
performWorkUntilDeadline @ scheduler.development.js:45
<WithRenderTrack(AIInterviewComponent)>
exports.jsxDEV @ react-jsx-dev-runtime.development.js:342
(익명) @ DynamicEngine.tsx:139
renderNodes @ DynamicEngine.tsx:31
(익명) @ DynamicEngine.tsx:110
renderNodes @ DynamicEngine.tsx:31
DynamicEngine @ DynamicEngine.tsx:172
react_stack_bottom_frame @ react-dom-client.development.js:28038
renderWithHooksAgain @ react-dom-client.development.js:8084
renderWithHooks @ react-dom-client.development.js:7996
updateFunctionComponent @ react-dom-client.development.js:10501
beginWork @ react-dom-client.development.js:12136
runWithFiberInDEV @ react-dom-client.development.js:986
performUnitOfWork @ react-dom-client.development.js:18997
workLoopSync @ react-dom-client.development.js:18825
renderRootSync @ react-dom-client.development.js:18806
performWorkOnRoot @ react-dom-client.development.js:17835
performWorkOnRootViaSchedulerTask @ react-dom-client.development.js:20384
performWorkUntilDeadline @ scheduler.development.js:45
<DynamicEngine>
exports.jsxDEV @ react-jsx-dev-runtime.development.js:342
CommonPage @ page.tsx:80
react_stack_bottom_frame @ react-dom-client.development.js:28038
renderWithHooksAgain @ react-dom-client.development.js:8084
renderWithHooks @ react-dom-client.development.js:7996
updateFunctionComponent @ react-dom-client.development.js:10501
beginWork @ react-dom-client.development.js:12136
runWithFiberInDEV @ react-dom-client.development.js:986
performUnitOfWork @ react-dom-client.development.js:18997
workLoopSync @ react-dom-client.development.js:18825
renderRootSync @ react-dom-client.development.js:18806
performWorkOnRoot @ react-dom-client.development.js:17835
performWorkOnRootViaSchedulerTask @ react-dom-client.development.js:20384
performWorkUntilDeadline @ scheduler.development.js:45
<CommonPage>
exports.jsx @ react-jsx-runtime.development.js:342
ClientPageRoot @ client-page.tsx:83
react_stack_bottom_frame @ react-dom-client.development.js:28038
renderWithHooksAgain @ react-dom-client.development.js:8084
renderWithHooks @ react-dom-client.development.js:7996
updateFunctionComponent @ react-dom-client.development.js:10501
beginWork @ react-dom-client.development.js:12085
runWithFiberInDEV @ react-dom-client.development.js:986
performUnitOfWork @ react-dom-client.development.js:18997
workLoopConcurrentByScheduler @ react-dom-client.development.js:18991
renderRootConcurrent @ react-dom-client.development.js:18973
performWorkOnRoot @ react-dom-client.development.js:17834
performWorkOnRootViaSchedulerTask @ react-dom-client.development.js:20384
performWorkUntilDeadline @ scheduler.development.js:45
"use client"
Function.all @ VM1117 <anonymous>:1
Function.all @ VM1117 <anonymous>:1
Function.all @ VM1117 <anonymous>:1
initializeElement @ react-server-dom-turbopack-client.browser.development.js:1932
"use server"
ResponseInstance @ react-server-dom-turbopack-client.browser.development.js:2767
createResponseFromOptions @ react-server-dom-turbopack-client.browser.development.js:4641
exports.createFromReadableStream @ react-server-dom-turbopack-client.browser.development.js:5045
module evaluation @ app-index.tsx:211
(익명) @ dev-base.ts:244
runModuleExecutionHooks @ dev-base.ts:278
instantiateModule @ dev-base.ts:238
getOrInstantiateModuleFromParent @ dev-base.ts:162
commonJsRequire @ runtime-utils.ts:389
(익명) @ app-next-turbopack.ts:11
(익명) @ app-bootstrap.ts:79
loadScriptsInSequence @ app-bootstrap.ts:23
appBootstrap @ app-bootstrap.ts:61
module evaluation @ app-next-turbopack.ts:10
(익명) @ dev-base.ts:244
runModuleExecutionHooks @ dev-base.ts:278
instantiateModule @ dev-base.ts:238
getOrInstantiateRuntimeModule @ dev-base.ts:128
registerChunk @ runtime-backend-dom.ts:57
await in registerChunk
registerChunk @ dev-base.ts:1149
(익명) @ dev-backend-dom.ts:126
(익명) @ dev-backend-dom.ts:126이 오류 이해하기
xhr.js:198  POST http://localhost:3000/api/ai/interview/resume/upload 500 (Internal Server Error)

http://localhost:3000/api/ai/interview/resume/upload
{
    "status": "error",
    "message": "서버 오류가 발생했습니다",
    "error": "S3Exception: The AWS Access Key Id you provided does not exist in our records. (Service: S3, Status Code: 403, Request ID: WTH84VN022G62GJQ, Extended Request ID: kwTjop+Hv20aCZg+PNeajbjUWl911FjqdkdIlxMKPun7N4e3mB8toHH7y8GI21eLjmRai6ev5yjcHlcb9CW9tdiVDx3CZz0T)",
    "timestamp": "2026-03-17T13:04:45.1169749",
    "path": "/api/ai/interview/resume/upload"
}



// [메모] PDF upload 에러 
http://localhost:3000/api/ai/interview/resume/upload

{
    "status": "success",
    "data": {
        "fileType": "pdf",
        "fileKey": "resume/1/5b670d32-5238-42ad-a972-6b435c99bc53.pdf"
    },
    "timestamp": "2026-03-17T15:46:26.9207923"
}


http://localhost:3000/api/ai/interview/start
{
    "resumeText": "",
    "resumeFileKey": "resume/1/5b670d32-5238-42ad-a972-6b435c99bc53.pdf",
    "language": "ko"
}

{
    "status": "error",
    "message": "서버 오류가 발생했습니다",
    "error": "FileNotFoundException: assets\\kdeliver-358f601d765c.json (지정된 경로를 찾을 수 없습니다)",
    "timestamp": "2026-03-17T15:46:32.7338286",
    "path": "/api/ai/interview/start"
}
AI_INTERVIEW_1600.png 참고

이미지 업로드
http://localhost:3000/api/ai/interview/resume/upload


{
    "status": "success",
    "data": {
        "fileType": "image",
        "fileKey": "resume/1/dcd22a4d-d591-4cb3-ad7c-a9abacd6848d.png"
    },
    "timestamp": "2026-03-17T15:52:47.0927907"
}

http://localhost:3000/api/ai/interview/start

{
    "resumeText": "",
    "resumeFileKey": "resume/1/dcd22a4d-d591-4cb3-ad7c-a9abacd6848d.png",
    "language": "ko"
}

이미지와 PDF에서 텍스트를 추출하는 로직이 필요함 
