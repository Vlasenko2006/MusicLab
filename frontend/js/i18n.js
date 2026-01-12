// i18n.js - Internationalization module

let currentLanguage = localStorage.getItem('language') || 'en';
let loadedTranslations = {};

// Fallback embedded translations for EN, DE, RU (legacy support)
const embeddedTranslations = {
    en: {
        nav: { contact: 'Contact', examples: 'Examples', about: 'About' },
        hero: { title: 'Create Unique Music with AI', subtitle: 'Upload two tracks and let AI fuse them into something new' },
        track: { 
            title1: 'Input Track A', 
            title2: 'Input Track B',
            noFile: 'No file uploaded',
            uploadText: 'Drag & drop audio file here or click to upload',
            uploadHint: 'MP3, WAV, OGG • Min. 16 seconds'
        },
        upload: { 
            title: 'Upload Your Tracks', 
            drop: 'Drag & drop audio file here or click to browse', 
            browse: 'Browse Files', 
            supported: 'Supported: MP3, WAV, M4A, OGG',
            instruction: 'Upload two tracks to start',
            instructionDetail: 'Select 16+ second audio files (MP3, WAV, OGG) for Track A and Track B. The generate button will appear automatically.'
        },
        player: { 
            selection: 'Time Selection', 
            start: 'Start', 
            end: 'End', 
            seconds: 'seconds', 
            preview: 'Preview Selection', 
            stop: 'Stop', 
            play: 'Play Entire Track', 
            volume: 'Volume',
            playSelection: 'Play Selection',
            clearTrack: 'Clear Track'
        },
        generate: { button: 'Generate New Track', processing: 'Processing...', ready: 'Upload both tracks to generate' },
        result: { title: 'Generated Result', play: 'Play Result', stop: 'Stop', download: 'Download', downloadTrack: 'Download Track' },
        contact: { label: 'Contact Email:' },
        about: { 
            title: 'About MusicLab',
            // ... (truncated for brevity - kept in app.js for now)
        },
        examples: { title: 'Music Generation Examples', close: 'Close' }
    }
    // DE and RU also available in app.js
};

async function loadTranslations(lang) {
    try {
        const response = await fetch(`/locales/${lang}.json`);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${lang}.json`);
        }
        const data = await response.json();
        console.log(`Loaded translations for ${lang} from JSON`);
        return data;
    } catch (error) {
        console.error(`Failed to load ${lang}.json, using embedded translations`, error);
        return embeddedTranslations[lang] || embeddedTranslations['en'];
    }
}

async function setLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('language', lang);
    
    // Load translations if not already loaded
    if (!loadedTranslations[lang]) {
        loadedTranslations[lang] = await loadTranslations(lang);
    }
    
    const langData = loadedTranslations[lang];
    
    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        const keys = key.split('.');
        let value = langData;
        
        for (const k of keys) {
            value = value[k];
            if (!value) break;
        }
        
        if (value) {
            element.textContent = value;
        }
    });
    
    // Update language button
    const langMap = { 
        en: 'EN', de: 'DE', ru: 'RU', fr: 'FR', 
        es: 'ES', pt: 'PT', ar: 'AR', zh: '中文' 
    };
    document.getElementById('current-lang').textContent = langMap[lang];
}

function initializeLanguageSelector() {
    const languageBtn = document.getElementById('language-btn');
    const languageMenu = document.getElementById('language-menu');
    
    languageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        languageMenu.style.display = languageMenu.style.display === 'block' ? 'none' : 'block';
    });
    
    document.addEventListener('click', (e) => {
        if (!languageBtn.contains(e.target)) {
            languageMenu.style.display = 'none';
        }
    });
    
    document.querySelectorAll('.language-option').forEach(option => {
        option.addEventListener('click', async (e) => {
            const lang = e.target.dataset.lang;
            await setLanguage(lang);
            languageMenu.style.display = 'none';
        });
    });
}

// Export functions for use in main app
export { currentLanguage, setLanguage, initializeLanguageSelector };
