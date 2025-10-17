// CSV dropzone wiring
(() => {
  const dz = document.getElementById('csvDrop');
  const input = document.getElementById('csvInput');
  const status = document.getElementById('status');

  if (!dz || !input) return;

  const setStatus = (msg) => { if (status) status.textContent = msg || ''; };

  dz.addEventListener('click', () => input.click());
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('dragover');
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  });
  input.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  });

  function handleFile(file){
    if (!/\.csv$/i.test(file.name)) {
      setStatus('Please provide a .csv file.');
      return;
    }
    setStatus(`Loaded: ${file.name} (${Math.round(file.size/1024)} KB)`);
    // TODO: parse or upload file here
    // Example: const reader = new FileReader(); reader.onload = (e)=>{ /* e.target.result */ }; reader.readAsText(file);
  }
})();
