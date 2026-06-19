// Animations globales GSAP + particules au chargement

document.addEventListener("DOMContentLoaded", () => {
  initParticles();
  animateNavbar();
  animateHero();
  animateCards();
});

function initParticles() {
  const container = document.getElementById("particles");
  if (!container) return;

  const colors = ["#f0b429", "#00853f", "#e31b23"];
  const count = 28;

  for (let i = 0; i < count; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    const size = Math.random() * 3 + 1;
    p.style.cssText = `
      width: ${size}px;
      height: ${size}px;
      left: ${Math.random() * 100}%;
      bottom: ${-Math.random() * 20}%;
      background: ${colors[Math.floor(Math.random() * colors.length)]};
      animation-duration: ${8 + Math.random() * 14}s;
      animation-delay: ${Math.random() * 8}s;
    `;
    container.appendChild(p);
  }
}

function animateNavbar() {
  gsap.from(".navbar", {
    y: -80,
    opacity: 0,
    duration: 0.8,
    ease: "power3.out"
  });
}

function animateHero() {
  const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
  tl.from("#heroTitle", { y: 50, opacity: 0, duration: 0.9, delay: 0.2 })
    .from("#heroSub",   { y: 30, opacity: 0, duration: 0.7 }, "-=0.5")
    .from(".badge",     { y: 20, opacity: 0, stagger: 0.1, duration: 0.5 }, "-=0.4");
}

function animateCards() {
  if (typeof ScrollTrigger !== "undefined") {
    gsap.registerPlugin(ScrollTrigger);
    gsap.utils.toArray(".glass-card, .stat-card, .about-card").forEach((el) => {
      gsap.from(el, {
        scrollTrigger: {
          trigger: el,
          start: "top 88%",
          toggleActions: "play none none none"
        },
        y: 40,
        opacity: 0,
        duration: 0.7,
        ease: "power3.out"
      });
    });
  }
}
