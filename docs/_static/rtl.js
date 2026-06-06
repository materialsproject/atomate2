// Set the document direction to RTL for right-to-left languages (e.g. Persian).
// Furo only emits <html lang="..."> without a `dir`, so set it here.
(function () {
  var rtl = ["fa", "ar", "he", "fa-IR", "ar-SA"];
  var lang = (document.documentElement.getAttribute("lang") || "").toLowerCase();
  if (rtl.indexOf(lang) !== -1 || lang.split("-")[0] === "fa") {
    document.documentElement.setAttribute("dir", "rtl");
  }
})();
