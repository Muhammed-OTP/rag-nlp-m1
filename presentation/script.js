(function () {
  const slides = Array.from(document.querySelectorAll(".slide"));
  const total = slides.length;
  let current = 0;

  const slideNum = document.getElementById("slideNum");
  const slideTotal = document.getElementById("slideTotal");
  const progressFill = document.getElementById("progressFill");
  const dotsContainer = document.getElementById("dots");
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");

  slideTotal.textContent = String(total).padStart(2, "0");

  // Build dot indicators
  const dots = slides.map((_, i) => {
    const dot = document.createElement("div");
    dot.className = "dot";
    dot.setAttribute("role", "button");
    dot.setAttribute("aria-label", "Aller à la diapositive " + (i + 1));
    dot.addEventListener("click", () => goTo(i));
    dotsContainer.appendChild(dot);
    return dot;
  });

  function render() {
    slides.forEach((slide, i) => {
      slide.classList.remove("active", "prev");
      if (i === current) {
        slide.classList.add("active");
      } else if (i < current) {
        slide.classList.add("prev");
      }
    });

    dots.forEach((dot, i) => dot.classList.toggle("active", i === current));

    slideNum.textContent = String(current + 1).padStart(2, "0");
    progressFill.style.width = ((current + 1) / total) * 100 + "%";

    prevBtn.classList.toggle("disabled", current === 0);
    nextBtn.classList.toggle("disabled", current === total - 1);
  }

  function goTo(index) {
    if (index < 0 || index >= total || index === current) return;
    current = index;
    render();
  }

  function next() { goTo(current + 1); }
  function prev() { goTo(current - 1); }

  prevBtn.addEventListener("click", prev);
  nextBtn.addEventListener("click", next);

  window.addEventListener("keydown", (e) => {
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
      case "PageDown":
      case " ":
        e.preventDefault();
        next();
        break;
      case "ArrowLeft":
      case "ArrowUp":
      case "PageUp":
        e.preventDefault();
        prev();
        break;
      case "Home":
        e.preventDefault();
        goTo(0);
        break;
      case "End":
        e.preventDefault();
        goTo(total - 1);
        break;
    }
  });

  // Touch swipe support
  let touchStartX = 0;
  window.addEventListener("touchstart", (e) => {
    touchStartX = e.changedTouches[0].screenX;
  });
  window.addEventListener("touchend", (e) => {
    const dx = e.changedTouches[0].screenX - touchStartX;
    if (Math.abs(dx) > 60) {
      if (dx < 0) next(); else prev();
    }
  });

  render();
})();
