// nav.js — floating nav scrollspy + BibTeX copy button
(function () {
  // ---------- Scrollspy for floating nav ----------
  const navLinks = document.querySelectorAll('.float-nav a[href^="#"]');
  const sections = Array.from(navLinks)
    .map(a => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  function onScroll() {
    const y = window.scrollY + 120;
    let activeIdx = 0;
    for (let i = 0; i < sections.length; i++) {
      if (sections[i].offsetTop <= y) activeIdx = i;
    }
    navLinks.forEach((a, i) => {
      a.classList.toggle('active', i === activeIdx);
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('load', onScroll);

  // ---------- BibTeX copy ----------
  const copyBtn = document.getElementById('copy-bibtex');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const text = document.getElementById('bibtex-content').innerText;
      navigator.clipboard.writeText(text).then(() => {
        const orig = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        copyBtn.classList.add('copied');
        setTimeout(() => {
          copyBtn.textContent = orig;
          copyBtn.classList.remove('copied');
        }, 1500);
      });
    });
  }
})();
