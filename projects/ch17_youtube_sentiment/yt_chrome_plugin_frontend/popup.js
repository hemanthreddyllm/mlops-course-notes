// YouTube Sentiment Insights: popup logic
// 1. read the active tab's URL → video id
// 2. fetch comments from the YouTube Data API v3
// 3. send them to the Flask backend → sentiments + charts
// 4. render the summary

const API_URL = "http://localhost:5001";            // after deployment: http://<ec2-public-ip>:8080
const YOUTUBE_API_KEY = "YOUR_YOUTUBE_DATA_API_KEY"; // Google Cloud → APIs & Services → YouTube Data API v3 → Credentials
const MAX_COMMENTS = 500;

const $ = (id) => document.getElementById(id);
const setStatus = (msg, isError = false) => { $("status").textContent = msg; $("status").className = "status" + (isError ? " error" : ""); };

function videoIdFrom(url) {
  const m = url && url.match(/^https:\/\/(?:www\.)?youtube\.com\/watch\?.*v=([\w-]{11})/);
  return m ? m[1] : null;
}

async function fetchComments(videoId) {
  if (window.__DEMO_COMMENTS__) return window.__DEMO_COMMENTS__;   // offline preview (dev_preview.html)
  const comments = [];
  let pageToken = "";
  while (comments.length < MAX_COMMENTS) {
    const url = `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}` +
                `&maxResults=100&pageToken=${pageToken}&key=${YOUTUBE_API_KEY}`;
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error?.message || `YouTube API error ${res.status}`);
    for (const item of data.items || []) {
      const s = item.snippet.topLevelComment.snippet;
      comments.push({ text: s.textOriginal, timestamp: s.publishedAt, authorId: s.authorChannelId?.value || s.authorDisplayName });
    }
    pageToken = data.nextPageToken;
    if (!pageToken) break;
  }
  return comments.slice(0, MAX_COMMENTS);
}

async function postJSON(path, body) {
  const res = await fetch(API_URL + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
  return res;
}

async function loadImage(path, body, imgId) {
  try {
    const blob = await (await postJSON(path, body)).blob();
    $(imgId).src = URL.createObjectURL(blob);
  } catch (e) {
    $(imgId).alt = `could not load (${e.message})`;
  }
}

function render(comments, predictions) {
  const counts = { "1": 0, "0": 0, "-1": 0 };
  predictions.forEach((p) => counts[String(p.sentiment)]++);
  const total = predictions.length;
  const words = comments.reduce((n, c) => n + c.text.split(/\s+/).filter(Boolean).length, 0);
  const avgSentiment = predictions.reduce((s, p) => s + p.sentiment, 0) / total;   // −1 … 1

  $("m-total").textContent = total;
  $("m-unique").textContent = new Set(comments.map((c) => c.authorId)).size;
  $("m-length").textContent = (words / total).toFixed(1);
  $("m-score").textContent = (((avgSentiment + 1) / 2) * 10).toFixed(1);

  const parts = [["positive", "1", "#1baf7a"], ["neutral", "0", "#9a9890"], ["negative", "-1", "#e34948"]];
  $("bar").innerHTML = parts.map(([, k, c]) => `<div style="width:${(100 * counts[k]) / total}%;background:${c}"></div>`).join("");
  $("bar-legend").innerHTML = parts.map(([n, k]) => `<span>${n} ${((100 * counts[k]) / total).toFixed(1)}%</span>`).join("");

  const cls = { "1": "pos", "0": "", "-1": "neg" };
  $("comments").innerHTML = predictions.slice(0, 25).map((p) => {
    const li = document.createElement("li");
    li.className = cls[String(p.sentiment)];
    li.textContent = p.comment;
    const small = document.createElement("small");
    small.textContent = `  · ${p.label}`;
    li.appendChild(small);
    return li.outerHTML;
  }).join("");

  $("results").hidden = false;
  return counts;
}

async function run() {
  try {
    const [tab] = await new Promise((resolve) => chrome.tabs.query({ active: true, currentWindow: true }, resolve));
    const videoId = videoIdFrom(tab && tab.url);
    if (!videoId) return setStatus("Open a YouTube video (youtube.com/watch?v=…) and click the extension again.", true);

    setStatus(`Video ${videoId}: fetching comments…`);
    const comments = await fetchComments(videoId);
    if (!comments.length) return setStatus("No comments found for this video.", true);

    setStatus(`Analysing ${comments.length} comments…`);
    const predictions = await (await postJSON("/predict_with_timestamps", {
      comments: comments.map((c) => ({ text: c.text, timestamp: c.timestamp })),
    })).json();

    const counts = render(comments, predictions);
    setStatus(`Done: ${comments.length} comments analysed.`);

    await Promise.all([
      loadImage("/generate_chart", { sentiment_counts: counts }, "chart"),
      loadImage("/generate_trend_graph", { sentiment_data: predictions }, "trend"),
      loadImage("/generate_wordcloud", { comments: comments.map((c) => c.text) }, "cloud"),
    ]);
  } catch (e) {
    setStatus(`Error: ${e.message}. Is the backend running at ${API_URL}?`, true);
  }
}

document.addEventListener("DOMContentLoaded", run);
