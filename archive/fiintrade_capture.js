// Paste this into the browser DevTools console (F12 -> Console) while on
// fiintrade.vn, logged into your trial account. It does NOT forge any
// requests or bypass their auth/signing — it just records whatever your
// own browsing already sends and receives, since the page's own JS
// computes the signed headers correctly for you automatically.
//
// After pasting this, browse normally: open the Money Flow / NDT charts,
// switch between investor types (Individual/Institution/Foreign/etc.),
// change tickers, and — most importantly — try to push the date range
// picker back before 2025 to see how far the underlying data actually
// goes. Every matching API response gets captured automatically.
//
// When done, run:  downloadFiinCapture()
// This saves everything captured so far as one JSON file.

(function () {
  window.__fiinCapture = window.__fiinCapture || [];

  // Broadened: catch ANY fiintrade API call, not just the ones we already know
  // by name — GetPriceData (technical.fiintrade.vn) turned out to carry the
  // full investor breakdown too, under a name we hadn't anticipated. Filtering
  // on "fiintrade" + "api-looking" (has a "?" query string) is safer than
  // trying to guess every relevant endpoint name up front.
  const MATCH = /fiintrade\.vn\/.+\?/i;

  const origFetch = window.fetch;
  window.fetch = async function (...args) {
    const res = await origFetch.apply(this, args);
    try {
      const url = typeof args[0] === "string" ? args[0] : args[0]?.url || "";
      if (MATCH.test(url)) {
        const clone = res.clone();
        clone.json().then((data) => {
          window.__fiinCapture.push({
            url,
            capturedAt: new Date().toISOString(),
            data,
          });
          console.log(
            `[fiin-capture] +1 (${window.__fiinCapture.length} total) <- ${url}`
          );
        }).catch(() => {});
      }
    } catch (e) {}
    return res;
  };

  // Some sites use XMLHttpRequest instead of fetch for these calls — cover both.
  const origOpen = XMLHttpRequest.prototype.open;
  const origSend = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    this.__fiinUrl = url;
    return origOpen.call(this, method, url, ...rest);
  };
  XMLHttpRequest.prototype.send = function (...args) {
    this.addEventListener("load", function () {
      try {
        if (MATCH.test(this.__fiinUrl || "")) {
          const data = JSON.parse(this.responseText);
          window.__fiinCapture.push({
            url: this.__fiinUrl,
            capturedAt: new Date().toISOString(),
            data,
          });
          console.log(
            `[fiin-capture] +1 (${window.__fiinCapture.length} total) <- ${this.__fiinUrl}`
          );
        }
      } catch (e) {}
    });
    return origSend.apply(this, args);
  };

  window.downloadFiinCapture = function () {
    const blob = new Blob([JSON.stringify(window.__fiinCapture, null, 2)], {
      type: "application/json",
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `fiintrade_capture_${Date.now()}.json`;
    a.click();
    console.log(`[fiin-capture] downloaded ${window.__fiinCapture.length} responses`);
  };

  console.log(
    "[fiin-capture] armed. Browse the Money Flow / NDT charts now (try different tickers, investor types, and pushing the date range before 2025). Run downloadFiinCapture() when done."
  );
})();
