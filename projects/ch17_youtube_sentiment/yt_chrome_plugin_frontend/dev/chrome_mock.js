// Offline preview only: pretend we are an extension popup opened on a YouTube video tab.
window.chrome = window.chrome || {};
chrome.tabs = { query: (_q, cb) => cb([{ url: "https://www.youtube.com/watch?v=w71RHxAWxaM" }]) };
