// ==UserScript==
// @name         FiinTrade Investor-Flow Harvester
// @namespace    fiintrade-harvest
// @version      2.5
// @description  Automated + gap-fill harvest of FiinTrade's investor-classification data via the real UI (no request forging). Multi-tab safe via localStorage locks.
// @author       you
// @match        https://fiintrade.vn/*
// @match        https://*.fiintrade.vn/*
// @grant        none
// @run-at       document-idle
// ==/UserScript==
//
// fiintrade_harvest.js — v2.5 with per-ticker auto-save, atomic state, and
// gap-fill mode (reads fiintrade_missing_dates.csv, fetches only the exact
// missing ranges instead of walking full history). Multi-tab: pick the CSV
// once in any tab via loadGapRanges() — it's cached in localStorage so every
// other tab's startGapHarvest() picks it up automatically. Per-ticker locks
// (also localStorage-based) mean tabs never double-fetch the same ticker.
//
// @grant none is required — this script patches window.fetch/XMLHttpRequest
// on the PAGE's own window to passively observe FiinTrade's real requests.
// Any other @grant value makes Tampermonkey run it in a sandboxed context
// where `window` is a proxy, not the actual page window, and the patch
// would silently do nothing.
(function () {
  // ---------- Tab ID ----------
  if (!window.__fiinTabId) {
    window.__fiinTabId = Math.random().toString(36).substring(2) + Date.now().toString(36);
  }
  const TAB_ID = window.__fiinTabId;

  // ---------- Event emitter ----------
  const captureListeners = [];

  function notifyCapture() {
    for (const cb of captureListeners) {
      try { cb(); } catch (e) {}
    }
  }

  // ---------- Network capture ----------
  window.__fiinCapture = window.__fiinCapture || [];
  const MATCH = /fiintrade\.vn\/.+\?/i;
  if (!window.__fiinPatched) {
    window.__fiinPatched = true;

    const origFetch = window.fetch;
    window.fetch = async function (...args) {
      const res = await origFetch.apply(this, args);
      try {
        const url = typeof args[0] === "string" ? args[0] : args[0]?.url || "";
        if (MATCH.test(url)) {
          const clone = res.clone();
          clone.json().then((data) => {
            window.__fiinCapture.push({ url, capturedAt: new Date().toISOString(), data });
            notifyCapture();
          }).catch(() => {});
        }
      } catch (e) {}
      return res;
    };

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
            window.__fiinCapture.push({ url: this.__fiinUrl, capturedAt: new Date().toISOString(), data });
            notifyCapture();
          }
        } catch (e) {}
      });
      return origSend.apply(this, args);
    };
  }

  // ---------- Config ----------
  const WINDOW_DAYS = 80;
  const WINDOW_END_DATE = new Date("2005-01-01");
  const MIN_DELAY_MS = 1200;
  const MAX_DELAY_MS = 2800;
  const TICKER_SWITCH_EXTRA_DELAY_MS = 1000;
  const EMPTY_WINDOWS_BEFORE_SKIP = 3;
  const MAX_APPLY_RETRIES = 3;
  const RETRY_DELAY_MS = 2000;
  const RESPONSE_WAIT_TIMEOUT_MS = 30000;
  const LOCK_TIMEOUT_MS = 10 * 60 * 1000;
  const LOCK_RENEW_INTERVAL_MS = 5 * 60 * 1000;

  const CLASSIFICATION_FIELDS = [
    'localIndividualBuyValue', 'localIndividualBuyVolume', 'localIndividualSellVolume',
    'localIndividualSellValue', 'localInstitutionalBuyVolume', 'localInstitutionalBuyValue',
    'localInstitutionalSellVolume', 'localInstitutionalSellValue',
    'proprietaryTotalBuyTradeVolume', 'proprietaryTotalBuyTradeValue',
    'proprietaryTotalSellTradeValue', 'proprietaryTotalSellTradeVolume',
    'netProprietaryMatchVolume', 'netProprietaryMatchValue',
    'netInstitutionMatchVolume', 'netInstitutionMatchValue'
  ];

  const STATE_KEY = 'fiintrade_harvest_state';
  const GAP_STATE_KEY = 'fiintrade_gap_harvest_state';

  let running = false;
  let stopRequested = false;
  let progress = { ticker: null, windowStart: null, windowsDone: 0, tickersRequested: [], tickersDone: [] };
  let currentLockTicker = null;

  // ---------- Helpers ----------
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function randomDelay() {
    return MIN_DELAY_MS + Math.random() * (MAX_DELAY_MS - MIN_DELAY_MS);
  }

  function fmtDDMMYYYY(d) {
    const dd = String(d.getDate()).padStart(2, "0");
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const yyyy = d.getFullYear();
    return `${dd}/${mm}/${yyyy}`;
  }

  function simulateClick(element) {
    if (!element) return;
    ["mousedown", "mouseup", "click"].forEach((eventType) => {
      element.dispatchEvent(new MouseEvent(eventType, {
        view: window, bubbles: true, cancelable: true, buttons: 1,
      }));
    });
  }

  async function simulateTyping(element, text, hitEnter = true) {
    element.focus();
    const isInput = element.tagName === "INPUT";
    if (isInput) {
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
      if (nativeInputValueSetter) nativeInputValueSetter.call(element, "");
      else element.value = "";
    } else {
      element.innerText = "";
    }
    element.dispatchEvent(new Event("input", { bubbles: true, cancelable: true }));
    await sleep(150);
    for (const char of text) {
      if (isInput) {
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
        if (nativeInputValueSetter) nativeInputValueSetter.call(element, element.value + char);
        else element.value += char;
      } else {
        element.innerText += char;
      }
      element.dispatchEvent(new Event("input", { bubbles: true, cancelable: true }));
      await sleep(50);
    }
    if (hitEnter) {
      element.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, key: "Enter", code: "Enter", keyCode: 13 }));
      element.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true, cancelable: true, key: "Enter", keyCode: 13 }));
    }
    element.blur();
  }

  async function waitForElement(selector, timeout = 10000) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      const el = document.querySelector(selector);
      if (el) return el;
      await sleep(200);
    }
    return null;
  }

  // ---------- UI actions ----------
  function clickInvestorClassificationTab() {
    const spans = document.querySelectorAll('li.nav-item.m-tabs__item a.nav-link span');
    for (const span of spans) {
      if (span.textContent.trim() === 'Phân loại nhà đầu tư') {
        const anchor = span.closest('a');
        if (anchor) {
          simulateClick(anchor);
          return true;
        }
      }
    }
    console.warn('[harvest] Could not find "Phân loại nhà đầu tư" tab');
    return false;
  }

  async function switchTicker(code) {
    const tickerDiv = document.querySelector('.ticker[contenteditable="true"]');
    if (!tickerDiv) throw new Error("Could not find .ticker[contenteditable] — is the widget open?");
    simulateClick(tickerDiv);
    await sleep(300);
    await simulateTyping(tickerDiv, code, true);
    await sleep(TICKER_SWITCH_EXTRA_DELAY_MS);
  }

  async function setDateRangeAndApply(fromDate, toDate) {
    const fromWrapper = document.querySelector(".wide.ml-10 a.narrow");
    if (fromWrapper) simulateClick(fromWrapper);

    const startDateInput = await waitForElement("#startDate");
    const endDateInput = await waitForElement("#endDate");
    if (!startDateInput || !endDateInput) {
      throw new Error("Date inputs (#startDate/#endDate) did not appear.");
    }

    simulateClick(startDateInput);
    await sleep(200);
    await simulateTyping(startDateInput, fmtDDMMYYYY(fromDate), false);
    await sleep(400);

    simulateClick(endDateInput);
    await sleep(200);
    await simulateTyping(endDateInput, fmtDDMMYYYY(toDate), true);

    const applyBtn = await waitForElement("button.apply.btn-apply-cus.apply-calendar");
    if (!applyBtn) throw new Error("Apply button did not appear.");
    await sleep(300);
    applyBtn.removeAttribute("disabled");
    simulateClick(applyBtn);
    applyBtn.click();
  }

  // ---------- Generic state management (shared shape, different keys) ----------
  function loadState(key) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (e) { return null; }
  }

  function saveState(key, state) {
    try {
      localStorage.setItem(key, JSON.stringify(state));
    } catch (e) {}
  }

  function clearState(key) {
    localStorage.removeItem(key);
  }

  async function addDoneTicker(key, ticker, requestedList) {
    let retries = 5;
    while (retries > 0) {
      const state = loadState(key);
      if (!state) {
        const newState = {
          tickersRequested: requestedList,
          tickersDone: [ticker],
          lastUpdated: new Date().toISOString()
        };
        saveState(key, newState);
        const verify = loadState(key);
        if (verify && verify.tickersDone && verify.tickersDone.includes(ticker)) {
          console.log(`[harvest] ${ticker}: added to done list (initialized).`);
          return true;
        }
        retries--;
        await sleep(100);
        continue;
      }

      const done = state.tickersDone || [];
      if (done.includes(ticker)) {
        return true;
      }
      const newDone = [...done, ticker];
      const newState = {
        tickersRequested: state.tickersRequested || requestedList,
        tickersDone: newDone,
        lastUpdated: new Date().toISOString()
      };
      saveState(key, newState);
      const verify = loadState(key);
      if (verify && verify.tickersDone && verify.tickersDone.includes(ticker)) {
        console.log(`[harvest] ${ticker}: added to done list.`);
        return true;
      }
      retries--;
      await sleep(100);
    }
    console.warn(`[harvest] ${ticker}: failed to add to done list after retries.`);
    return false;
  }

  // ---------- Locks ----------
  function getLockKey(ticker) { return `fiintrade_lock_${ticker}`; }

  function getLock(ticker) {
    const raw = localStorage.getItem(getLockKey(ticker));
    if (!raw) return null;
    try { return JSON.parse(raw); } catch { return null; }
  }

  function setLock(ticker, tabId, timestamp) {
    localStorage.setItem(getLockKey(ticker), JSON.stringify({ tabId, timestamp }));
  }

  function releaseLock(ticker) {
    localStorage.removeItem(getLockKey(ticker));
  }

  function isLockExpired(lock) {
    if (!lock) return true;
    return (Date.now() - lock.timestamp) > LOCK_TIMEOUT_MS;
  }

  function tryAcquireLock(ticker) {
    let lock = getLock(ticker);
    if (lock && !isLockExpired(lock) && lock.tabId !== TAB_ID) {
      return false;
    }
    setLock(ticker, TAB_ID, Date.now());
    const newLock = getLock(ticker);
    if (newLock && newLock.tabId === TAB_ID) {
      return true;
    }
    return false;
  }

  function renewLock(ticker) {
    const lock = getLock(ticker);
    if (lock && lock.tabId === TAB_ID) {
      setLock(ticker, TAB_ID, Date.now());
      return true;
    }
    return false;
  }

  // ---------- Data emptiness ----------
  function getInvestorDataCount(responseData) {
    if (!responseData) return 0;
    let items = null;
    const possibleProps = ['items', 'data', 'rows', 'records'];
    for (const prop of possibleProps) {
      if (Array.isArray(responseData[prop])) {
        items = responseData[prop];
        break;
      }
    }
    if (!items && Array.isArray(responseData)) {
      items = responseData;
    }
    if (!items || items.length === 0) return 0;
    let count = 0;
    for (const item of items) {
      if (!item || typeof item !== 'object') continue;
      let hasClassification = false;
      for (const field of CLASSIFICATION_FIELDS) {
        if (item[field] !== null && item[field] !== undefined) {
          hasClassification = true;
          break;
        }
      }
      if (hasClassification) count++;
    }
    return count;
  }

  // ---------- Download helper ----------
  function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ---------- Apply one date range with retry + response wait (shared) ----------
  async function applyRangeWithRetry(code, windowStart, windowEnd, lockKey) {
    const beforeCount = window.__fiinCapture.length;
    let responseReceived = false;
    let applyRetries = 0;

    while (!responseReceived && applyRetries < MAX_APPLY_RETRIES && !stopRequested) {
      const lockCheck = getLock(lockKey);
      if (!lockCheck || lockCheck.tabId !== TAB_ID) {
        console.warn(`[harvest] ${code}: lock lost during apply — stopping.`);
        stopRequested = true;
        break;
      }

      try {
        if (applyRetries > 0) {
          console.log(`[harvest] Retry #${applyRetries} for ${code} — re-clicking Apply`);
        }
        await setDateRangeAndApply(windowStart, windowEnd);

        let listener = null;
        const waitForResponse = new Promise((resolve, reject) => {
          let resolved = false;
          const timeoutId = setTimeout(() => {
            if (!resolved) { resolved = true; reject(new Error('Response wait timeout')); }
          }, RESPONSE_WAIT_TIMEOUT_MS);

          listener = () => {
            if (window.__fiinCapture.length > beforeCount) {
              resolved = true;
              clearTimeout(timeoutId);
              resolve();
            }
          };

          if (window.__fiinCapture.length > beforeCount) {
            clearTimeout(timeoutId);
            resolve();
          } else {
            captureListeners.push(listener);
          }
        });

        try {
          await waitForResponse;
          responseReceived = true;
        } catch (e) {
          console.warn(`[harvest] ${code}: attempt ${applyRetries + 1} — timeout`);
          if (listener) {
            const idx = captureListeners.indexOf(listener);
            if (idx > -1) captureListeners.splice(idx, 1);
          }
          await sleep(RETRY_DELAY_MS);
        }
      } catch (e) {
        console.warn(`[harvest] ${code}: Apply error — ${e.message}`);
        await sleep(RETRY_DELAY_MS);
      }
      applyRetries++;
    }

    if (!responseReceived) return { got: 0, investorCount: null };
    const newOnes = window.__fiinCapture.slice(beforeCount);
    const lastResponse = newOnes[newOnes.length - 1].data;
    return { got: newOnes.length, investorCount: getInvestorDataCount(lastResponse) };
  }

  // ---------- Full-history harvest (walks backward from today) ----------
  async function harvestTicker(code) {
    if (!tryAcquireLock(code)) {
      console.log(`[harvest] ${code}: lock already held – skipping.`);
      return false;
    }
    currentLockTicker = code;
    console.log(`[harvest] ${code}: lock acquired (tab ${TAB_ID})`);

    const renewInterval = setInterval(() => {
      if (stopRequested || !currentLockTicker) {
        clearInterval(renewInterval);
        return;
      }
      if (!renewLock(currentLockTicker)) {
        console.warn(`[harvest] ${code}: lost lock – stopping.`);
        stopRequested = true;
        clearInterval(renewInterval);
      }
    }, LOCK_RENEW_INTERVAL_MS);

    const captureStartIndex = window.__fiinCapture.length;
    let completed = false;

    try {
      await switchTicker(code);
      clickInvestorClassificationTab();
      await sleep(400);

      let windowEnd = new Date();
      let consecutiveEmpty = 0;

      while (!stopRequested && windowEnd > WINDOW_END_DATE) {
        const lock = getLock(code);
        if (!lock || lock.tabId !== TAB_ID) {
          console.warn(`[harvest] ${code}: lock lost – stopping.`);
          break;
        }

        const windowStart = new Date(windowEnd);
        windowStart.setDate(windowStart.getDate() - WINDOW_DAYS);
        progress.ticker = code;
        progress.windowStart = windowStart.toISOString().slice(0, 10);

        console.log(`[harvest] ${code} 🌐 Window: ${windowStart.toISOString().slice(0, 10)} → ${windowEnd.toISOString().slice(0, 10)}`);

        const { got, investorCount } = await applyRangeWithRetry(code, windowStart, windowEnd, code);
        if (investorCount === null) {
          console.warn(`[harvest] ${code} ${progress.windowStart}: failed after ${MAX_APPLY_RETRIES} attempts – skipping window.`);
          windowEnd = windowStart;
          continue;
        }

        progress.windowsDone += 1;
        console.log(`[harvest] ${code}  window ending ${windowEnd.toISOString().slice(0, 10)}  (+${got} resp, investorRecords=${investorCount})`);

        if (investorCount === 0) {
          consecutiveEmpty += 1;
          if (consecutiveEmpty >= EMPTY_WINDOWS_BEFORE_SKIP) {
            console.log(`[harvest] ${code}: ${consecutiveEmpty} consecutive empty windows — moving to next ticker.`);
            break;
          }
        } else {
          consecutiveEmpty = 0;
        }

        windowEnd = windowStart;
        if (stopRequested) break;
        await sleep(randomDelay());
      }

      if (!stopRequested) completed = true;
    } catch (e) {
      console.error(`[harvest] fatal error on ${code}:`, e);
    } finally {
      clearInterval(renewInterval);
      releaseLock(code);
      currentLockTicker = null;
    }

    if (completed) {
      const tickerData = window.__fiinCapture.slice(captureStartIndex);
      if (tickerData.length) {
        downloadJSON(tickerData, `fiintrade_${code}_${Date.now()}.json`);
        console.log(`[harvest] ${code}: auto-saved ${tickerData.length} responses.`);
      } else {
        console.log(`[harvest] ${code}: no new responses captured.`);
      }
    }

    return completed;
  }

  async function runHarvest(tickers, resume) {
    let savedState = loadState(STATE_KEY);
    let requestedTickers = tickers.slice();
    let doneTickers = [];

    if (resume && savedState) {
      if (savedState.tickersRequested && savedState.tickersRequested.length > 0) {
        requestedTickers = savedState.tickersRequested;
        doneTickers = savedState.tickersDone || [];
        doneTickers = doneTickers.filter(t => requestedTickers.includes(t));
        console.log(`[harvest] Resuming: ${doneTickers.length} done out of ${requestedTickers.length}.`);
      } else {
        requestedTickers = tickers.slice();
        doneTickers = savedState.tickersDone || [];
        console.log(`[harvest] Using passed list, ${doneTickers.length} tickers already done.`);
      }
    } else {
      console.log(`[harvest] Fresh start.`);
      clearState(STATE_KEY);
    }

    const remaining = requestedTickers.filter(t => !doneTickers.includes(t));
    if (remaining.length === 0) {
      console.log("[harvest] All tickers already done.");
      return;
    }

    progress.tickersRequested = requestedTickers;
    progress.tickersDone = doneTickers;
    progress.windowsDone = 0;
    progress.ticker = null;
    progress.windowStart = null;
    saveState(STATE_KEY, { tickersRequested: requestedTickers, tickersDone: doneTickers, lastUpdated: new Date().toISOString() });

    running = true;
    stopRequested = false;

    console.log(`[harvest] Starting (tab ${TAB_ID}) – ${remaining.length} tickers left.`);

    while (!stopRequested) {
      const state = loadState(STATE_KEY);
      if (!state) break;
      const done = state.tickersDone || [];
      const requested = state.tickersRequested || [];
      const remainingNow = requested.filter(t => !done.includes(t));

      if (remainingNow.length === 0) {
        console.log("[harvest] All tickers done. Stopping.");
        break;
      }

      let claimed = false;
      for (const t of remainingNow) {
        if (stopRequested) break;
        if (tryAcquireLock(t)) {
          console.log(`[harvest] ${t}: claimed by tab ${TAB_ID}`);
          claimed = true;
          const success = await harvestTicker(t);
          if (success) {
            const added = await addDoneTicker(STATE_KEY, t, requested);
            if (added) {
              const newState = loadState(STATE_KEY);
              if (newState) {
                progress.tickersDone = newState.tickersDone || [];
                progress.tickersRequested = newState.tickersRequested || requested;
              }
            } else {
              console.warn(`[harvest] ${t}: could not add to done – will retry later.`);
            }
          } else {
            console.log(`[harvest] ${t}: harvest failed – will retry.`);
          }
          break;
        }
      }

      if (!claimed) {
        console.log("[harvest] No ticker available – waiting...");
        await sleep(5000);
      }
    }

    running = false;
    if (!stopRequested) {
      console.log("[harvest] All done! Clearing saved state.");
      clearState(STATE_KEY);
    } else {
      console.log("[harvest] Stopped early. State preserved.");
    }
    console.log(`[harvest] ${progress.tickersDone.length}/${progress.tickersRequested.length} tickers completed.`);
  }

  // ==========================================================================
  // GAP-FILL MODE — reads fiintrade_missing_dates.csv (ticker,from_date,
  // to_date,missing_days — produced by archive/find_fiintrade_gaps.py) and
  // fetches ONLY those exact ranges per ticker, instead of walking the full
  // history. Same lock/resume/auto-save machinery as the full-history mode,
  // just a different range source and a separate state key so the two modes
  // don't collide if you run both across tabs.
  // ==========================================================================

  let gapRangesByTicker = null; // Map<ticker, [{from: Date, to: Date}, ...]>
  const GAP_CSV_CACHE_KEY = 'fiintrade_gap_ranges_csv'; // raw CSV text, shared across tabs via localStorage

  // `new Date(string)` is dangerously ambiguous for slash-separated dates —
  // it always assumes US MM/DD/YYYY regardless of what the string actually
  // means, so a DD/MM/YYYY value like "10/05/2021" (10 May) silently becomes
  // October 5th. The source CSV (archive/find_fiintrade_gaps.py) writes
  // ISO YYYY-MM-DD, but it's easy to end up with DD/MM/YYYY instead (e.g.
  // after opening/resaving the file in Excel, which reformats date-looking
  // cells to the system locale). Parse both explicitly by field position —
  // never hand a slash-string to `new Date()` directly.
  function parseDateFlexible(str) {
    const s = str.trim();
    let m = s.match(/^(\d{4})-(\d{2})-(\d{2})$/); // ISO: YYYY-MM-DD
    if (m) return new Date(+m[1], +m[2] - 1, +m[3]);
    m = s.match(/^(\d{2})\/(\d{2})\/(\d{4})$/); // DD/MM/YYYY (Excel-locale form)
    if (m) return new Date(+m[3], +m[2] - 1, +m[1]);
    return new Date(NaN); // unrecognized format — treat as invalid, don't guess
  }

  function parseCSV(text) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    const header = lines[0].split(",").map((h) => h.trim());
    const idx = {
      ticker: header.indexOf("ticker"),
      from: header.indexOf("from_date"),
      to: header.indexOf("to_date"),
    };
    if (idx.ticker === -1 || idx.from === -1 || idx.to === -1) {
      throw new Error("CSV must have columns: ticker,from_date,to_date,missing_days");
    }
    const map = new Map();
    let skipped = 0;
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(",");
      const ticker = cols[idx.ticker].trim();
      let from = parseDateFlexible(cols[idx.from]);
      let to = parseDateFlexible(cols[idx.to]);
      if (!ticker || isNaN(from) || isNaN(to)) { skipped++; continue; }
      if (from > to) [from, to] = [to, from]; // defensive: never emit a reversed range
      const list = map.get(ticker) || [];
      list.push({ from, to });
      map.set(ticker, list);
    }
    if (skipped > 0) {
      console.warn(`[harvest] parseCSV: skipped ${skipped} row(s) with unrecognized/invalid dates.`);
    }
    return map;
  }

  // Opens a native file picker, parses the selected CSV, stores it for
  // startGapHarvest() to use AND caches the raw text in localStorage so
  // every other tab on this origin can pick it up automatically — you only
  // need to run this once, in ONE tab; every other tab's startGapHarvest()
  // auto-loads from the cache (see loadGapRangesFromCache below).
  window.loadGapRanges = function () {
    return new Promise((resolve, reject) => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".csv";
      input.style.display = "none";
      document.body.appendChild(input);
      input.addEventListener("change", () => {
        const file = input.files[0];
        document.body.removeChild(input);
        if (!file) { reject(new Error("No file selected")); return; }
        const reader = new FileReader();
        reader.onload = () => {
          try {
            gapRangesByTicker = parseCSV(reader.result);
            try {
              localStorage.setItem(GAP_CSV_CACHE_KEY, reader.result);
            } catch (e) {
              console.warn("[harvest] Could not cache CSV in localStorage (too large?) — other tabs will need to load it themselves.", e);
            }
            const totalRanges = [...gapRangesByTicker.values()].reduce((s, l) => s + l.length, 0);
            console.log(`[harvest] Loaded ${gapRangesByTicker.size} tickers, ${totalRanges} ranges from ${file.name} (cached for other tabs).`);
            resolve(gapRangesByTicker);
          } catch (e) {
            reject(e);
          }
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
      });
      input.click();
    });
  };

  // Reads the CSV cached by loadGapRanges() in another tab. Returns false
  // (no throw) if nothing has been cached yet, so callers can fall back to
  // prompting for a file picker instead.
  window.loadGapRangesFromCache = function () {
    const cached = localStorage.getItem(GAP_CSV_CACHE_KEY);
    if (!cached) return false;
    gapRangesByTicker = parseCSV(cached);
    const totalRanges = [...gapRangesByTicker.values()].reduce((s, l) => s + l.length, 0);
    console.log(`[harvest] Loaded ${gapRangesByTicker.size} tickers, ${totalRanges} ranges from cache (shared by another tab).`);
    return true;
  };

  async function harvestTickerGaps(code, ranges) {
    if (!tryAcquireLock(code)) {
      console.log(`[harvest] ${code}: lock already held – skipping.`);
      return false;
    }
    currentLockTicker = code;
    console.log(`[harvest] ${code}: lock acquired (tab ${TAB_ID}), ${ranges.length} gap range(s) to fetch.`);

    const renewInterval = setInterval(() => {
      if (stopRequested || !currentLockTicker) {
        clearInterval(renewInterval);
        return;
      }
      if (!renewLock(currentLockTicker)) {
        console.warn(`[harvest] ${code}: lost lock – stopping.`);
        stopRequested = true;
        clearInterval(renewInterval);
      }
    }, LOCK_RENEW_INTERVAL_MS);

    const captureStartIndex = window.__fiinCapture.length;
    let completed = false;

    try {
      await switchTicker(code);
      clickInvestorClassificationTab();
      await sleep(400);

      for (const { from, to } of ranges) {
        if (stopRequested) break;
        const lock = getLock(code);
        if (!lock || lock.tabId !== TAB_ID) {
          console.warn(`[harvest] ${code}: lock lost – stopping.`);
          break;
        }

        progress.ticker = code;
        progress.windowStart = from.toISOString().slice(0, 10);
        console.log(`[harvest] ${code} 🌐 Gap range: ${from.toISOString().slice(0, 10)} → ${to.toISOString().slice(0, 10)}`);

        const { got, investorCount } = await applyRangeWithRetry(code, from, to, code);
        progress.windowsDone += 1;
        if (investorCount === null) {
          console.warn(`[harvest] ${code} ${progress.windowStart}: failed after ${MAX_APPLY_RETRIES} attempts – skipping range.`);
        } else {
          console.log(`[harvest] ${code}  range ending ${to.toISOString().slice(0, 10)}  (+${got} resp, investorRecords=${investorCount})`);
        }

        if (stopRequested) break;
        await sleep(randomDelay());
      }

      if (!stopRequested) completed = true;
    } catch (e) {
      console.error(`[harvest] fatal error on ${code}:`, e);
    } finally {
      clearInterval(renewInterval);
      releaseLock(code);
      currentLockTicker = null;
    }

    if (completed) {
      const tickerData = window.__fiinCapture.slice(captureStartIndex);
      if (tickerData.length) {
        downloadJSON(tickerData, `fiintrade_gapfill_${code}_${Date.now()}.json`);
        console.log(`[harvest] ${code}: auto-saved ${tickerData.length} responses.`);
      } else {
        console.log(`[harvest] ${code}: no new responses captured.`);
      }
    }

    return completed;
  }

  async function runGapHarvest(resume) {
    if (!gapRangesByTicker || gapRangesByTicker.size === 0) {
      // This tab hasn't loaded ranges itself — try the shared cache another
      // tab may have populated via loadGapRanges() before prompting again.
      if (!window.loadGapRangesFromCache()) {
        console.warn("[harvest] No gap ranges loaded — run loadGapRanges() first (in this tab or any other tab on this origin) and pick fiintrade_missing_dates.csv.");
        return;
      }
    }
    const allTickers = [...gapRangesByTicker.keys()];

    let savedState = loadState(GAP_STATE_KEY);
    let requestedTickers = allTickers;
    let doneTickers = [];

    if (resume && savedState && savedState.tickersRequested && savedState.tickersRequested.length > 0) {
      requestedTickers = savedState.tickersRequested;
      doneTickers = (savedState.tickersDone || []).filter((t) => requestedTickers.includes(t));
      console.log(`[harvest] Resuming gap-fill: ${doneTickers.length} done out of ${requestedTickers.length}.`);
    } else {
      console.log("[harvest] Gap-fill fresh start.");
      clearState(GAP_STATE_KEY);
    }

    const remaining = requestedTickers.filter((t) => !doneTickers.includes(t));
    if (remaining.length === 0) {
      console.log("[harvest] All gap-fill tickers already done.");
      return;
    }

    progress.tickersRequested = requestedTickers;
    progress.tickersDone = doneTickers;
    progress.windowsDone = 0;
    saveState(GAP_STATE_KEY, { tickersRequested: requestedTickers, tickersDone: doneTickers, lastUpdated: new Date().toISOString() });

    running = true;
    stopRequested = false;
    console.log(`[harvest] Starting gap-fill (tab ${TAB_ID}) – ${remaining.length} tickers left.`);

    while (!stopRequested) {
      const state = loadState(GAP_STATE_KEY);
      if (!state) break;
      const done = state.tickersDone || [];
      const requested = state.tickersRequested || [];
      const remainingNow = requested.filter((t) => !done.includes(t));

      if (remainingNow.length === 0) {
        console.log("[harvest] All gap-fill tickers done. Stopping.");
        break;
      }

      let claimed = false;
      for (const t of remainingNow) {
        if (stopRequested) break;
        if (tryAcquireLock(t)) {
          console.log(`[harvest] ${t}: claimed by tab ${TAB_ID}`);
          claimed = true;
          const ranges = gapRangesByTicker.get(t) || [];
          const success = await harvestTickerGaps(t, ranges);
          if (success) {
            await addDoneTicker(GAP_STATE_KEY, t, requested);
          } else {
            console.log(`[harvest] ${t}: gap-fill failed – will retry.`);
          }
          break;
        }
      }

      if (!claimed) {
        console.log("[harvest] No ticker available – waiting...");
        await sleep(5000);
      }
    }

    running = false;
    if (!stopRequested) {
      console.log("[harvest] Gap-fill all done! Clearing saved state.");
      clearState(GAP_STATE_KEY);
    } else {
      console.log("[harvest] Gap-fill stopped early. State preserved.");
    }
  }

  window.startGapHarvest = async function (resume = true) {
    if (running) {
      console.warn("[harvest] Already running.");
      return;
    }
    await runGapHarvest(resume);
  };

  window.gapHarvestStatus = function () {
    const state = loadState(GAP_STATE_KEY);
    const done = state ? state.tickersDone : [];
    const requested = state ? state.tickersRequested : [];
    const remaining = requested.filter((t) => !done.includes(t));
    console.log(JSON.stringify({
      running,
      tabId: TAB_ID,
      loadedTickers: gapRangesByTicker ? gapRangesByTicker.size : 0,
      currentTicker: progress.ticker,
      tickersDone: done.length,
      tickersTotal: requested.length,
      tickersRemaining: remaining.length,
      totalCaptured: window.__fiinCapture.length,
    }, null, 1));
  };

  // ---------- Public API (full-history mode) ----------
  window.startHarvest = async function (tickers, resume = true) {
    if (running) {
      console.warn("[harvest] Already running.");
      return;
    }
    await runHarvest(tickers, resume);
  };

  window.stopHarvest = function () {
    stopRequested = true;
    if (currentLockTicker) {
      releaseLock(currentLockTicker);
      currentLockTicker = null;
    }
    console.log("[harvest] Stop requested.");
  };

  window.harvestStatus = function () {
    const state = loadState(STATE_KEY);
    const done = state ? state.tickersDone : [];
    const requested = state ? state.tickersRequested : [];
    const remaining = requested.filter(t => !done.includes(t));
    console.log(JSON.stringify({
      running,
      tabId: TAB_ID,
      currentTicker: progress.ticker,
      currentWindowStart: progress.windowStart,
      windowsDone: progress.windowsDone,
      tickersDone: done.length,
      tickersTotal: requested.length,
      tickersRemaining: remaining.length,
      totalCaptured: window.__fiinCapture.length,
    }, null, 1));
  };

  window.downloadHarvest = function () {
    downloadJSON(window.__fiinCapture, `fiintrade_harvest_${Date.now()}.json`);
    console.log(`[harvest] Downloaded ${window.__fiinCapture.length} total responses.`);
  };

  window.clearHarvestState = function () {
    clearState(STATE_KEY);
    clearState(GAP_STATE_KEY);
    const keys = Object.keys(localStorage);
    for (const k of keys) {
      if (k.startsWith('fiintrade_lock_')) localStorage.removeItem(k);
    }
    console.log("[harvest] State and locks cleared.");
  };

  window.showLastResponseStructure = function () {
    const last = window.__fiinCapture[window.__fiinCapture.length - 1];
    if (!last) { console.log("No responses yet."); return; }
    console.log("Last response keys:", Object.keys(last.data));
    console.log("Sample data:", last.data);
  };

  // ==========================================================================
  // GUI PANEL — small floating control box, top-left, so you don't need the
  // console open for routine use. Every button just calls the same
  // window.* functions already exposed above; console commands still work
  // identically alongside it.
  // ==========================================================================

  function buildPanel() {
    if (document.getElementById("fiin-harvest-panel")) return; // don't double-inject on re-run

    const panel = document.createElement("div");
    panel.id = "fiin-harvest-panel";
    Object.assign(panel.style, {
      position: "fixed", top: "10px", left: "10px", zIndex: 999999,
      background: "#1e1e1e", color: "#eee", fontFamily: "monospace", fontSize: "11px",
      padding: "8px", borderRadius: "6px", boxShadow: "0 2px 10px rgba(0,0,0,0.5)",
      width: "230px", userSelect: "none",
    });

    const title = document.createElement("div");
    title.textContent = "FiinTrade Harvester v2.5";
    Object.assign(title.style, { fontWeight: "bold", marginBottom: "6px", cursor: "move" });
    panel.appendChild(title);

    // Drag-to-move by the title bar
    (function makeDraggable() {
      let dragging = false, offX = 0, offY = 0;
      title.addEventListener("mousedown", (e) => {
        dragging = true;
        offX = e.clientX - panel.offsetLeft;
        offY = e.clientY - panel.offsetTop;
      });
      document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        panel.style.left = `${e.clientX - offX}px`;
        panel.style.top = `${e.clientY - offY}px`;
      });
      document.addEventListener("mouseup", () => { dragging = false; });
    })();

    const status = document.createElement("div");
    Object.assign(status.style, { marginBottom: "6px", lineHeight: "1.5", whiteSpace: "pre-wrap" });
    panel.appendChild(status);

    function addButton(label, onClick) {
      const btn = document.createElement("button");
      btn.textContent = label;
      Object.assign(btn.style, {
        display: "block", width: "100%", marginBottom: "4px", padding: "4px",
        background: "#333", color: "#eee", border: "1px solid #555", borderRadius: "4px",
        cursor: "pointer", fontFamily: "monospace", fontSize: "11px",
      });
      btn.addEventListener("mouseenter", () => { btn.style.background = "#444"; });
      btn.addEventListener("mouseleave", () => { btn.style.background = "#333"; });
      btn.addEventListener("click", async () => {
        try {
          await onClick();
        } catch (e) {
          console.error("[harvest][panel]", e);
        }
      });
      panel.appendChild(btn);
      return btn;
    }

    addButton("📂 Load gap CSV", () => window.loadGapRanges());
    addButton("▶️ Start gap-fill harvest", () => window.startGapHarvest());
    addButton("▶️ Start full-history harvest", () => {
      const list = window.prompt("Comma-separated tickers, e.g. VCB,HPG,FPT");
      if (!list) return;
      return window.startHarvest(list.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean));
    });
    addButton("⏸ Stop", () => window.stopHarvest());
    addButton("💾 Download all captured", () => window.downloadHarvest());
    addButton("🗑 Clear state & locks", () => window.clearHarvestState());

    document.body.appendChild(panel);

    function refreshStatus() {
      const gapState = loadState(GAP_STATE_KEY);
      const fullState = loadState(STATE_KEY);
      const mode = gapRangesByTicker ? "gap-fill" : (fullState ? "full-history" : "idle");
      const state = gapState || fullState;
      const done = state ? (state.tickersDone || []).length : 0;
      const total = state ? (state.tickersRequested || []).length : 0;
      status.textContent =
        `mode: ${mode}\n` +
        `running: ${running}\n` +
        `ticker: ${progress.ticker || "-"}\n` +
        `progress: ${done}/${total} tickers\n` +
        `captured: ${window.__fiinCapture.length}`;
    }
    refreshStatus();
    setInterval(refreshStatus, 2000);
  }

  buildPanel();

  console.log("[harvest] ✅ v2.5 loaded. Tab ID:", TAB_ID);
  console.log("  GUI panel added top-left — drag by its title bar to move it.");
  console.log("  Full history:  startHarvest(['list'])  harvestStatus()");
  console.log("  Gap-fill:      in ONE tab: await loadGapRanges() (picks fiintrade_missing_dates.csv)");
  console.log("                 in EVERY tab (incl. that one): startGapHarvest()  gapHarvestStatus()");
  console.log("                 (other tabs auto-load the CSV from localStorage cache — no need to re-pick it)");
  console.log("  Shared:        stopHarvest()  downloadHarvest()  clearHarvestState()");
})();
