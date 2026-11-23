(() => {
  const cfg = window.MAIGYM_DOCS || { root: "", rawMd: "", slug: "index.html" };

  /* ---- Theme Toggle (synced with site/ via maigym.theme) ---- */
  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "day" ? "night" : "day";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("maigym.theme", next); } catch (e) {}
    });
  }

  /* ---- Mobile Drawer ---- */
  const navBtn = document.getElementById("mobile-nav-btn");
  const sidebar = document.getElementById("sidebar");
  const backdrop = document.getElementById("drawer-backdrop");
  function setDrawer(open) {
    if (!sidebar || !backdrop || !navBtn) return;
    sidebar.classList.toggle("open", open);
    backdrop.hidden = !open;
    navBtn.setAttribute("aria-expanded", String(open));
  }
  if (navBtn) navBtn.addEventListener("click", () => setDrawer(!sidebar.classList.contains("open")));
  if (backdrop) backdrop.addEventListener("click", () => setDrawer(false));
  const activeSideLink = sidebar?.querySelector(".side-link.active");
  if (activeSideLink) activeSideLink.scrollIntoView({ block: "nearest" });

  /* ---- Copy Code Buttons ---- */
  function flashCopied(btn, labelText) {
    btn.classList.add("copied");
    const span = btn.querySelector("span");
    const prev = span ? span.textContent : "";
    if (span) span.textContent = labelText || "Copied";
    setTimeout(() => {
      btn.classList.remove("copied");
      if (span) span.textContent = prev;
    }, 1800);
  }

  async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
  }

  document.querySelectorAll("[data-copy-code]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const code = btn.closest(".code-block")?.querySelector("pre code");
      if (!code) return;
      try {
        await copyText(code.textContent || "");
        flashCopied(btn, "Copied");
      } catch (e) {}
    });
  });

  /* ---- Copy Page as Markdown (Mintlify signature) ---- */
  const copyPageBtn = document.getElementById("copy-page-btn");
  if (copyPageBtn && cfg.rawMd) {
    copyPageBtn.addEventListener("click", async () => {
      try {
        const res = await fetch(cfg.rawMd);
        if (!res.ok) throw new Error("fetch failed");
        const md = await res.text();
        await copyText(md);
        flashCopied(copyPageBtn, "Copied MD");
      } catch (e) {
        flashCopied(copyPageBtn, "Unavailable");
      }
    });
  }

  /* ---- Back to top ---- */
  const backTop = document.getElementById("back-to-top");
  if (backTop) {
    backTop.addEventListener("click", (e) => {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  /* ---- Right TOC Scroll Spy ---- */
  const tocLinks = Array.from(document.querySelectorAll("#toc-list .toc-link"));
  if (tocLinks.length) {
    const targets = tocLinks
      .map((link) => {
        const id = (link.getAttribute("href") || "").slice(1);
        return { link, el: id ? document.getElementById(id) : null };
      })
      .filter((x) => x.el);

    function updateToc() {
      const scrollY = window.scrollY + 140;
      let current = targets[0];
      for (const item of targets) {
        if (item.el.offsetTop <= scrollY) current = item;
        else break;
      }
      tocLinks.forEach((l) => l.classList.remove("active"));
      if (current) current.link.classList.add("active");
    }

    let ticking = false;
    window.addEventListener("scroll", () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(() => {
          updateToc();
          ticking = false;
        });
      }
    }, { passive: true });
    updateToc();
  }

  /* ---- Image Zoom Lightbox ---- */
  const imgDialog = document.getElementById("img-dialog");
  const imgDialogSrc = document.getElementById("img-dialog-src");
  const imgDialogClose = document.getElementById("img-dialog-close");
  if (imgDialog && imgDialogSrc) {
    document.querySelectorAll("img[data-zoomable]").forEach((img) => {
      img.addEventListener("click", () => {
        imgDialogSrc.src = img.currentSrc || img.src;
        imgDialogSrc.alt = img.alt || "";
        imgDialog.showModal();
      });
    });
    if (imgDialogClose) imgDialogClose.addEventListener("click", () => imgDialog.close());
    imgDialog.addEventListener("click", (e) => {
      if (e.target === imgDialog) imgDialog.close();
    });
  }

  /* ---- Instant Cmd+K Search Modal ---- */
  const searchTrigger = document.getElementById("search-trigger");
  const searchDialog = document.getElementById("search-dialog");
  const searchInput = document.getElementById("search-input");
  const searchResults = document.getElementById("search-results");
  const searchClose = document.getElementById("search-close");

  let indexPromise = null;
  let selectedIdx = 0;
  let renderedItems = [];

  function loadIndex() {
    if (!indexPromise) {
      indexPromise = fetch(`${cfg.root}search-index.json`)
        .then((r) => (r.ok ? r.json() : []))
        .catch(() => []);
    }
    return indexPromise;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function highlightMatch(text, terms) {
    let out = escapeHtml(text);
    for (const t of terms) {
      if (t.length < 2) continue;
      const re = new RegExp(`(${t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "ig");
      out = out.replace(re, "<mark>$1</mark>");
    }
    return out;
  }

  function snippetAround(content, terms) {
    const lower = content.toLowerCase();
    let pos = -1;
    for (const t of terms) {
      const p = lower.indexOf(t);
      if (p !== -1 && (pos === -1 || p < pos)) pos = p;
    }
    if (pos === -1) return content.slice(0, 150);
    const start = Math.max(0, pos - 48);
    const end = Math.min(content.length, pos + 110);
    return (start > 0 ? "..." : "") + content.slice(start, end) + (end < content.length ? "..." : "");
  }

  async function runSearch(query) {
    const docs = await loadIndex();
    const q = query.trim().toLowerCase();
    const terms = q.split(/\s+/).filter(Boolean);

    let matches = [];
    if (!terms.length) {
      matches = docs.slice(0, 8).map((d) => ({
        href: `${cfg.root}${d.slug}`,
        title: d.title,
        group: d.group,
        snippet: d.description,
      }));
    } else {
      for (const d of docs) {
        let score = 0;
        const titleL = d.title.toLowerCase();
        const descL = d.description.toLowerCase();
        const contentL = d.content.toLowerCase();
        let bestHeading = null;

        for (const t of terms) {
          if (titleL.includes(t)) score += 10;
          if (descL.includes(t)) score += 5;
          if (contentL.includes(t)) score += 2;
          for (const h of d.headings || []) {
            if (h.text.toLowerCase().includes(t)) {
              score += 7;
              if (!bestHeading) bestHeading = h;
            }
          }
        }
        if (score > 0) {
          matches.push({
            score,
            href: `${cfg.root}${d.slug}${bestHeading ? "#" + bestHeading.id : ""}`,
            title: bestHeading ? `${d.title} › ${bestHeading.text}` : d.title,
            group: d.group,
            snippet: snippetAround(d.content, terms),
          });
        }
      }
      matches.sort((a, b) => b.score - a.score);
      matches = matches.slice(0, 10);
    }

    selectedIdx = 0;
    renderedItems = matches;
    if (!matches.length) {
      searchResults.innerHTML = `<div class="search-empty">No results for "${escapeHtml(query)}"</div>`;
      return;
    }
    searchResults.innerHTML = matches
      .map(
        (m, i) => `
      <a class="search-item${i === 0 ? " selected" : ""}" href="${escapeHtml(m.href)}" data-idx="${i}" role="option">
        <div class="search-item-head">
          <span class="search-item-title">${highlightMatch(m.title, terms)}</span>
          <span class="search-item-group">${escapeHtml(m.group)}</span>
        </div>
        <div class="search-item-snippet">${highlightMatch(m.snippet, terms)}</div>
      </a>`
      )
      .join("");
  }

  function openSearch() {
    if (!searchDialog || !searchInput) return;
    if (!searchDialog.open) searchDialog.showModal();
    searchInput.value = "";
    searchInput.focus();
    runSearch("");
  }

  if (searchTrigger) searchTrigger.addEventListener("click", openSearch);
  if (searchClose) searchClose.addEventListener("click", () => searchDialog.close());
  if (searchDialog) {
    searchDialog.addEventListener("click", (e) => {
      if (e.target === searchDialog) searchDialog.close();
    });
  }
  if (searchInput) {
    searchInput.addEventListener("input", () => runSearch(searchInput.value));
    searchInput.addEventListener("keydown", (e) => {
      const items = searchResults.querySelectorAll(".search-item");
      if (!items.length) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        selectedIdx = (selectedIdx + 1) % items.length;
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        selectedIdx = (selectedIdx - 1 + items.length) % items.length;
      } else if (e.key === "Enter" && renderedItems[selectedIdx]) {
        e.preventDefault();
        window.location.href = renderedItems[selectedIdx].href;
        return;
      } else {
        return;
      }
      items.forEach((el, idx) => el.classList.toggle("selected", idx === selectedIdx));
      items[selectedIdx]?.scrollIntoView({ block: "nearest" });
    });
  }

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      openSearch();
    } else if (
      e.key === "/" &&
      !searchDialog?.open &&
      document.activeElement?.tagName !== "INPUT" &&
      document.activeElement?.tagName !== "TEXTAREA"
    ) {
      e.preventDefault();
      openSearch();
    } else if (e.key === "Escape") {
      setDrawer(false);
    }
  });
})();
