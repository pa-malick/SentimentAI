// Logique de la page d'analyse : requete API, affichage du resultat, animations

(function () {
  const textarea    = document.getElementById("inputText");
  const charCount   = document.getElementById("charCount");
  const analyzeBtn  = document.getElementById("analyzeBtn");
  const resultBox   = document.getElementById("resultContainer");
  const errorBox    = document.getElementById("errorContainer");
  const modelBtns   = document.querySelectorAll(".model-btn");
  const exampleCards = document.querySelectorAll(".example-card");

  let selectedModel = "logistic";

  // Compteur de caracteres
  textarea.addEventListener("input", () => {
    charCount.textContent = textarea.value.length;
  });

  // Selection du modele
  modelBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      modelBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      selectedModel = btn.dataset.model;

      gsap.from(btn, { scale: 0.95, duration: 0.2, ease: "back.out(2)" });
    });
  });

  // Exemples rapides
  exampleCards.forEach((card) => {
    card.addEventListener("click", () => {
      textarea.value = card.dataset.text;
      charCount.textContent = textarea.value.length;
      textarea.scrollIntoView({ behavior: "smooth", block: "center" });
      gsap.from(textarea, { borderColor: "#f0b429", duration: 0.5 });
    });
  });

  // Bouton analyser
  analyzeBtn.addEventListener("click", analyzeText);

  textarea.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") analyzeText();
  });

  async function analyzeText() {
    const text = textarea.value.trim();
    if (!text || text.length < 5) {
      showError("Entrez au moins 5 caracteres avant d'analyser.");
      return;
    }

    hideResult();
    hideError();
    setLoading(true);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model: selectedModel })
      });

      const data = await response.json();

      if (!response.ok) {
        showError(data.error || "Erreur serveur. Verifie que les modeles sont entraines.");
        return;
      }

      displayResult(data);
    } catch (err) {
      showError("Impossible de contacter le serveur. Verifie que Flask est lance.");
    } finally {
      setLoading(false);
    }
  }

  function displayResult(data) {
    const isPositive = data.label === 1;
    const icon       = isPositive ? "&#x1F600;" : "&#x1F614;";
    const cssClass   = isPositive ? "positive" : "negative";
    const confidence = data.confidence;

    document.getElementById("sentimentIcon").innerHTML = icon;
    document.getElementById("sentimentText").textContent = data.sentiment;
    document.getElementById("sentimentText").className = `sentiment-text ${cssClass}`;
    document.getElementById("resultModel").textContent  = data.model;
    document.getElementById("confidenceValue").textContent = `${confidence}%`;

    const pills = document.getElementById("resultPills");
    pills.innerHTML = `
      <span class="badge">${data.model}</span>
      <span class="badge">Confiance : ${confidence}%</span>
      <span class="badge">${data.sentiment}</span>
    `;

    resultBox.style.display = "block";

    gsap.from(resultBox, { opacity: 0, y: 20, duration: 0.6, ease: "power3.out" });

    // Animation de la barre de confiance
    setTimeout(() => {
      document.getElementById("confidenceFill").style.width = `${confidence}%`;
    }, 100);
  }

  function showError(msg) {
    document.getElementById("errorMsg").textContent = msg;
    errorBox.style.display = "flex";
    gsap.from(errorBox, { opacity: 0, x: -10, duration: 0.4 });
  }

  function hideResult() { resultBox.style.display = "none"; }
  function hideError()  { errorBox.style.display  = "none"; }

  function setLoading(state) {
    if (state) {
      analyzeBtn.classList.add("loading");
      analyzeBtn.querySelector(".btn-text").textContent = "Analyse en cours...";
      gsap.to(analyzeBtn, { opacity: 0.7, duration: 0.2 });
    } else {
      analyzeBtn.classList.remove("loading");
      analyzeBtn.querySelector(".btn-text").textContent = "Analyser le sentiment";
      gsap.to(analyzeBtn, { opacity: 1, duration: 0.2 });
    }
  }
})();
