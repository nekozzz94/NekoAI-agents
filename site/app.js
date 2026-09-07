// app.js — HR AI Agent docs renderer

const renderer = {
  link(href, title, text) {
    // rewrite .md → .html links
    if (href && href.endsWith('.md')) {
      href = href.replace(/\.md$/, '.html').replace(/^\.\//, '');
    }
    const t = title ? ` title="${title}"` : '';
    return `<a href="${href}"${t}>${text}</a>`;
  },

  code(code, lang) {
    const validLang = lang && hljs.getLanguage(lang) ? lang : 'text';
    let highlighted;
    try {
      highlighted = hljs.highlight(code, { language: validLang, ignoreIllegals: true }).value;
    } catch {
      highlighted = code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
    const badge = lang ? `<span class="code-lang">${lang}</span>` : '<span class="code-lang">text</span>';
    return (
      `<div class="code-wrap">` +
      `<pre>` +
      `<div class="code-header">${badge}<button class="copy-btn" onclick="copyCode(this)">copy</button></div>` +
      `<code class="hljs language-${validLang}">${highlighted}</code>` +
      `</pre>` +
      `</div>`
    );
  }
};

marked.use({ renderer, gfm: true, breaks: false });

function slugify(text) {
  return text.toLowerCase()
    .replace(/[^\w\s-]/g, '').replace(/\s+/g, '-').replace(/-+/g, '-').trim();
}

function buildToc() {
  const content = document.getElementById('content');
  const tocList = document.getElementById('toc-list');
  const toc     = document.getElementById('toc');
  if (!content || !tocList || !toc) return;

  const headings = Array.from(content.querySelectorAll('h2, h3'));
  if (headings.length < 2) { toc.hidden = true; return; }

  const slugCount = {};
  headings.forEach(h => {
    const text = h.textContent.trim();
    const base = slugify(text);
    slugCount[base] = (slugCount[base] || 0) + 1;
    h.id = slugCount[base] > 1 ? `${base}-${slugCount[base]}` : base;

    const li = document.createElement('li');
    li.className = `toc-item toc-${h.tagName.toLowerCase()}`;
    const a = document.createElement('a');
    a.href  = `#${h.id}`;
    a.className = 'toc-link';
    a.textContent = text;
    li.appendChild(a);
    tocList.appendChild(li);
  });

  function updateActive() {
    const scrollY = window.scrollY + 88;
    let active = headings[0];
    for (const h of headings) {
      if (h.offsetTop <= scrollY) active = h; else break;
    }
    tocList.querySelectorAll('.toc-link').forEach(a => {
      a.classList.toggle('toc-active', a.getAttribute('href') === `#${active.id}`);
    });
  }

  window.addEventListener('scroll', updateActive, { passive: true });
  updateActive();
}

const COLLAPSE_THRESHOLD = 320; // px — blocks taller than this get a toggle

function initCollapsible() {
  document.querySelectorAll('#content .code-wrap').forEach(wrap => {
    const pre = wrap.querySelector('pre');
    if (!pre || pre.scrollHeight <= COLLAPSE_THRESHOLD) return;

    wrap.classList.add('collapsible');
    pre.classList.add('collapsed');

    const btn = document.createElement('button');
    btn.className = 'expand-btn';
    btn.textContent = 'Show more';
    btn.addEventListener('click', () => {
      const nowCollapsed = pre.classList.toggle('collapsed');
      btn.textContent = nowCollapsed ? 'Show more' : 'Show less';
    });
    wrap.appendChild(btn);
  });
}

function renderPage() {
  const src = document.getElementById('md-source');
  const el  = document.getElementById('content');
  if (!src || !el) return;
  el.innerHTML = marked.parse(src.textContent);
  buildToc();
  initCollapsible();
}

function copyCode(btn) {
  const code = btn.closest('pre').querySelector('code');
  navigator.clipboard.writeText(code.innerText).then(() => {
    btn.textContent = 'copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('copied'); }, 2000);
  });
}
