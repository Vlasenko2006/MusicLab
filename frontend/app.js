// MusicLab - Main Application JavaScript
// Handles audio upload, waveform visualization, pattern selection, and generation

// Translations
const translations = {
    en: {
        nav: { contact: 'Contact', examples: 'Examples', about: 'About' },
        hero: { title: 'Create Unique Music with AI', subtitle: 'Upload two tracks and let AI blend them into something new' },
        track: { 
            title1: 'Input Track A', 
            title2: 'Input Track B',
            noFile: 'No file uploaded',
            uploadText: 'Drop audio file here or click to upload',
            uploadHint: 'MP3, WAV, OGG • Min 16 seconds'
        },
        upload: { 
            title: 'Upload Your Tracks', 
            drop: 'Drop audio file here or click to browse', 
            browse: 'Browse Files', 
            supported: 'Supported: MP3, WAV, M4A, OGG',
            instruction: 'Upload Two Tracks to Begin',
            instructionDetail: 'Select 16+ second audio files (MP3, WAV, OGG) for Track A and Track B. The Generate button will appear automatically.'
        },
        player: { 
            selection: 'Time Selection', 
            start: 'Start', 
            end: 'End', 
            seconds: 'seconds', 
            preview: 'Preview Selection', 
            stop: 'Stop', 
            play: 'Play Full Track', 
            volume: 'Volume',
            playSelection: 'Play Selection',
            clearTrack: 'Clear Track'
        },
        generate: { button: 'Generate New Track', processing: 'Processing...', ready: 'Upload both tracks to generate' },
        result: { title: 'Generated Result', play: 'Play Result', stop: 'Stop', download: 'Download', downloadTrack: 'Download Track' },
        contact: { label: 'Contact Email:' },
        about: { 
            title: 'About MusicLab',
            welcome: 'Welcome to MusicLab',
            whatIs: 'What is MusicLab?',
            whatIsDesc: 'MusicLab is an innovative AI-powered music generation platform that creates entirely new music by fusing patterns from two input tracks. Think of it as a creative music mixer that learns from your favorite songs and generates something unique!',
            howWorks: 'How Does It Work?',
            step1Title: 'Upload Your Tracks',
            step1Desc: 'Choose two audio files (MP3, WAV, M4A, or FLAC) - Track A and Track B. These can be any genre or style!',
            step2Title: 'Select Time Windows',
            step2Desc: 'Use the sliding controls to choose a 16-second segment from each track. This is where you pick the patterns you want to fuse.',
            step3Title: 'AI Magic Happens',
            step3Desc: 'Our SimpleTransformer model (16.7M parameters) analyzes both segments using EnCodec audio encoding and generates new musical patterns.',
            step4Title: 'Download Your Creation',
            step4Desc: 'Listen to the generated music and download it as a WAV file!',
            close: 'Close',
            interface: 'Understanding the Interface',
            interfaceStep1: 'Step 1: Upload Your Music',
            interfaceStep2: 'Step 2: Select Your Patterns',
            interfaceStep3: 'Step 3: Generate and Download',
            dragUpload: 'Drag/Upload Your Music Samples:',
            dragUploadDesc: 'You can either drag and drop audio files directly onto the upload areas or click to browse your computer. Each track area accepts MP3, WAV, OGG, or FLAC files with a minimum duration of 16 seconds.',
            trackAB: 'Track A and Track B:',
            trackABDesc: 'Upload two different music tracks that you want to fuse together. The AI will learn patterns from both tracks to create something entirely new.',
            chatbot: 'Musical Chatbot:',
            chatbotDesc: 'Notice the floating eighth note icon in the bottom-right corner? That\'s MusicNote, your AI music assistant! Click it anytime to get help with pattern selection, music theory tips, or questions about how MusicLab works.',
            waveform: 'Waveform Display:',
            waveformDesc: 'Once uploaded, you\'ll see a visual waveform representation of your entire track. This helps you identify interesting sections with strong beats, melodies, or textures.',
            slidingWindow: 'Sliding Window Control:',
            slidingWindowDesc: 'Use the slider below each waveform to select exactly which 16-second segment you want to use. The blue highlighted area on the waveform shows your current selection. You can see the start and end times (in seconds) displayed below the waveform.',
            playEntire: 'Play Entire Pattern:',
            playEntireDesc: 'Click the large play button below the waveform to listen to your complete uploaded track from start to finish. This helps you explore the full song before making your selection.',
            previewSel: 'Preview Selection:',
            previewSelDesc: 'Use the "Preview Selection" button to listen to only the 16-second segment you\'ve chosen with the slider. This is the exact portion that will be sent to the AI for music generation.',
            clearTrackTitle: 'Clear Track:',
            clearTrackDesc: 'If you want to start over, click "Clear Track" to remove the uploaded file and begin again.',
            durationInd: 'Duration Indicator:',
            durationIndDesc: 'The blue "16.0 sec" indicator confirms that you\'ve selected a valid 16-second pattern. Both tracks must have a selection before you can generate music.',
            playGen: 'Play Generated Music:',
            playGenDesc: 'Once generation is complete, you\'ll see a new waveform showing your AI-created music. Click the play button to listen to the result. The player shows the current playback time and allows you to adjust volume.',
            downloadTitle: 'Download Track:',
            downloadDesc: 'Love what you created? Click the "Download Track" button to save your generated music as a high-quality WAV file to your computer. You can use this in your own projects!',
            createAnother: 'Create Another:',
            createAnotherDesc: 'Want to try different patterns or tracks? Click "Create Another" to return to the beginning and start a new fusion experiment.',
            assistantTitle: 'MusicNote Assistant:',
            assistantDesc: 'The chatbot is always available to help! You can ask questions like "What patterns work well together?" or "How can I improve my results?" MusicNote has deep knowledge of music theory and can guide you toward better creative combinations.',
            why16: 'Why 16 Seconds?',
            why16Desc: 'We use 16-second segments for several technical and creative reasons:',
            why16Pattern: 'Pattern Recognition:',
            why16PatternDesc: '16 seconds is long enough for the AI to recognize meaningful musical patterns like rhythm, melody, and harmony.',
            why16Efficiency: 'Model Efficiency:',
            why16EfficiencyDesc: 'Our EnCodec encoder works optimally with this duration, balancing quality and processing speed.',
            why16Memory: 'Memory Constraints:',
            why16MemoryDesc: 'Shorter segments allow for efficient processing on standard hardware without requiring expensive GPU resources.',
            why16Structure: 'Musical Structure:',
            why16StructureDesc: '16 seconds typically captures 1-2 musical phrases or bars (depending on tempo), which is enough context for creative fusion.',
            why16Focus: 'Focus:',
            why16FocusDesc: 'It forces you to choose the most interesting part of each track, leading to more creative results!',
            tips: 'Tips for Best Results',
            tipRhythm: 'Complementary Rhythms',
            tipRhythmDesc: 'Choose one track with a strong beat and another with a compelling melody. This creates interesting interplay between rhythm and melodic elements.',
            tipHarmony: 'Harmonic Compatibility',
            tipHarmonyDesc: 'Tracks in the same or related keys (e.g., C major and A minor) work best together. This ensures harmonies blend smoothly without clashing.',
            tipEnergy: 'Energy Balance',
            tipEnergyDesc: 'Mix high-energy sections with calmer ones for interesting dynamic contrast. This adds emotional depth and keeps listeners engaged.',
            tipTimbral: 'Timbral Variety',
            tipTimbralDesc: 'Different instruments (e.g., guitar + piano) create richer fusions. Combining diverse timbres adds texture and complexity to the output.',
            tipSilence: 'Avoid Silence',
            tipSilenceDesc: 'Choose segments with actual musical content, not silent or very quiet parts. The AI needs sufficient information to generate compelling patterns.',
            tipExperiment: 'Experiment!',
            tipExperimentDesc: 'Try unexpected combinations - sometimes the most creative results come from unlikely pairings. Don\'t be afraid to mix different genres!',
            technical: 'Technical Details',
            needHelp: 'Need Help?',
            needHelpDesc: 'Click the floating eighth note icon in the bottom-right corner to chat with MusicNote, our AI music assistant! MusicNote can help you with:',
            helpUnderstand: 'Understanding how MusicLab\'s AI model works',
            helpChoose: 'Choosing the best pattern combinations for creative results',
            helpTheory: 'Music theory concepts and production techniques',
            helpTechnical: 'Technical questions about audio formats and processing',
            helpInspiration: 'Creative inspiration and suggestions for your next fusion',
            helpTrouble: 'Troubleshooting any issues you encounter',
            helpConversation: 'MusicNote maintains conversation history throughout your session, so you can have a natural dialogue about your music creation process. The chatbot is knowledgeable, friendly, and always ready to assist you in making the most creative music possible!'
        },
        examples: { title: 'Example Outputs', close: 'Close' }
    },
    de: {
        nav: { contact: 'Kontakt', examples: 'Beispiele', about: 'Über uns' },
        hero: { title: 'Erstelle einzigartige Musik mit KI', subtitle: 'Lade zwei Tracks hoch und lass die KI sie zu etwas Neuem verschmelzen' },
        track: { 
            title1: 'Eingangs-Track A', 
            title2: 'Eingangs-Track B',
            noFile: 'Keine Datei hochgeladen',
            uploadText: 'Audiodatei hier ablegen oder klicken zum Hochladen',
            uploadHint: 'MP3, WAV, OGG • Min. 16 Sekunden'
        },
        upload: { 
            title: 'Lade deine Tracks hoch', 
            drop: 'Audiodatei hier ablegen oder klicken zum Durchsuchen', 
            browse: 'Dateien durchsuchen', 
            supported: 'Unterstützt: MP3, WAV, M4A, OGG',
            instruction: 'Lade zwei Tracks hoch um zu beginnen',
            instructionDetail: 'Wähle Audiodateien mit 16+ Sekunden (MP3, WAV, OGG) für Track A und Track B. Der Generieren-Button erscheint automatisch.'
        },
        player: { 
            selection: 'Zeitauswahl', 
            start: 'Start', 
            end: 'Ende', 
            seconds: 'Sekunden', 
            preview: 'Auswahl vorhören', 
            stop: 'Stopp', 
            play: 'Ganzen Track abspielen', 
            volume: 'Lautstärke',
            playSelection: 'Auswahl abspielen',
            clearTrack: 'Track löschen'
        },
        generate: { button: 'Neuen Track generieren', processing: 'Verarbeitung...', ready: 'Lade beide Tracks hoch zum Generieren' },
        result: { title: 'Generiertes Ergebnis', play: 'Ergebnis abspielen', stop: 'Stopp', download: 'Herunterladen', downloadTrack: 'Track herunterladen' },
        contact: { label: 'Kontakt E-Mail:' },
        about: { 
            title: 'Über MusicLab',
            welcome: 'Willkommen bei MusicLab',
            whatIs: 'Was ist MusicLab?',
            whatIsDesc: 'MusicLab ist eine innovative KI-gestützte Musikgenerierungsplattform, die völlig neue Musik erstellt, indem sie Muster aus zwei Eingangstracks fusioniert. Betrachten Sie es als kreativen Musikmixer, der von Ihren Lieblingsliedern lernt und etwas Einzigartiges generiert!',
            howWorks: 'Wie funktioniert es?',
            step1Title: 'Lade deine Tracks hoch',
            step1Desc: 'Wähle zwei Audiodateien (MP3, WAV, M4A oder FLAC) - Track A und Track B. Diese können jedes Genre oder jeden Stil haben!',
            step2Title: 'Wähle Zeitfenster',
            step2Desc: 'Verwende die Schieberegler, um ein 16-Sekunden-Segment von jedem Track auszuwählen. Hier wählst du die Muster aus, die du fusionieren möchtest.',
            step3Title: 'KI-Magie geschieht',
            step3Desc: 'Unser SimpleTransformer-Modell (16,7M Parameter) analysiert beide Segmente mit EnCodec-Audiocodierung und generiert neue musikalische Muster.',
            step4Title: 'Lade deine Kreation herunter',
            step4Desc: 'Höre dir die generierte Musik an und lade sie als WAV-Datei herunter!',
            close: 'Schließen',
            interface: 'Die Benutzeroberfläche verstehen',
            interfaceStep1: 'Schritt 1: Lade deine Musik hoch',
            interfaceStep2: 'Schritt 2: Wähle deine Muster',
            interfaceStep3: 'Schritt 3: Generieren und Herunterladen',
            dragUpload: 'Drag & Drop/Hochladen:',
            dragUploadDesc: 'Du kannst Audiodateien direkt in die Upload-Bereiche ziehen oder auf den Computer durchsuchen klicken. Jeder Track-Bereich akzeptiert MP3, WAV, OGG oder FLAC-Dateien mit einer Mindestdauer von 16 Sekunden.',
            trackAB: 'Track A und Track B:',
            trackABDesc: 'Lade zwei verschiedene Musiktracks hoch, die du fusionieren möchtest. Die KI lernt Muster aus beiden Tracks, um etwas völlig Neues zu erstellen.',
            chatbot: 'Musik-Chatbot:',
            chatbotDesc: 'Beachte das schwebende Achtelnoten-Symbol in der unteren rechten Ecke? Das ist MusicNote, dein KI-Musikassistent! Klicke jederzeit darauf, um Hilfe bei der Musterauswahl, Musiktheorie-Tipps oder Fragen zur Funktionsweise von MusicLab zu erhalten.',
            waveform: 'Wellenform-Anzeige:',
            waveformDesc: 'Nach dem Hochladen siehst du eine visuelle Wellenform-Darstellung deines gesamten Tracks. Dies hilft dir, interessante Abschnitte mit starken Beats, Melodien oder Texturen zu identifizieren.',
            slidingWindow: 'Schieberegler-Steuerung:',
            slidingWindowDesc: 'Verwende den Schieberegler unter jeder Wellenform, um genau das 16-Sekunden-Segment auszuwählen, das du verwenden möchtest. Der blau hervorgehobene Bereich auf der Wellenform zeigt deine aktuelle Auswahl. Du kannst die Start- und Endzeiten (in Sekunden) unter der Wellenform sehen.',
            playEntire: 'Gesamtes Muster abspielen:',
            playEntireDesc: 'Klicke auf die große Play-Taste unter der Wellenform, um deinen vollständig hochgeladenen Track von Anfang bis Ende anzuhören. Dies hilft dir, das gesamte Lied zu erkunden, bevor du deine Auswahl triffst.',
            previewSel: 'Auswahl vorhören:',
            previewSelDesc: 'Verwende die "Auswahl vorhören"-Taste, um nur das 16-Sekunden-Segment anzuhören, das du mit dem Schieberegler ausgewählt hast. Dies ist der genaue Teil, der zur KI-Musikgenerierung gesendet wird.',
            clearTrackTitle: 'Track löschen:',
            clearTrackDesc: 'Wenn du von vorne beginnen möchtest, klicke auf "Track löschen", um die hochgeladene Datei zu entfernen und erneut zu beginnen.',
            durationInd: 'Dauer-Anzeige:',
            durationIndDesc: 'Die blaue "16.0 sec"-Anzeige bestätigt, dass du ein gültiges 16-Sekunden-Muster ausgewählt hast. Beide Tracks müssen eine Auswahl haben, bevor du Musik generieren kannst.',
            playGen: 'Generierte Musik abspielen:',
            playGenDesc: 'Nach Abschluss der Generierung siehst du eine neue Wellenform, die deine von der KI erstellte Musik zeigt. Klicke auf die Play-Taste, um das Ergebnis anzuhören. Der Player zeigt die aktuelle Wiedergabezeit und ermöglicht die Lautstärkeregelung.',
            downloadTitle: 'Track herunterladen:',
            downloadDesc: 'Liebst du, was du erstellt hast? Klicke auf die "Track herunterladen"-Taste, um deine generierte Musik als hochwertige WAV-Datei auf deinen Computer zu speichern. Du kannst sie in deinen eigenen Projekten verwenden!',
            createAnother: 'Weitere erstellen:',
            createAnotherDesc: 'Möchtest du verschiedene Muster oder Tracks ausprobieren? Klicke auf "Weitere erstellen", um zum Anfang zurückzukehren und ein neues Fusions-Experiment zu starten.',
            assistantTitle: 'MusicNote-Assistent:',
            assistantDesc: 'Der Chatbot ist immer verfügbar, um zu helfen! Du kannst Fragen stellen wie "Welche Muster funktionieren gut zusammen?" oder "Wie kann ich meine Ergebnisse verbessern?" MusicNote hat tiefes Wissen über Musiktheorie und kann dich zu besseren kreativen Kombinationen führen.',
            why16: 'Warum 16 Sekunden?',
            why16Desc: 'Wir verwenden 16-Sekunden-Segmente aus mehreren technischen und kreativen Gründen:',
            why16Pattern: 'Mustererkennung:',
            why16PatternDesc: '16 Sekunden sind lang genug, damit die KI bedeutsame musikalische Muster wie Rhythmus, Melodie und Harmonie erkennen kann.',
            why16Efficiency: 'Modell-Effizienz:',
            why16EfficiencyDesc: 'Unser EnCodec-Encoder funktioniert mit dieser Dauer optimal und gleicht Qualität und Verarbeitungsgeschwindigkeit aus.',
            why16Memory: 'Speicher-Einschränkungen:',
            why16MemoryDesc: 'Kürzere Segmente ermöglichen eine effiziente Verarbeitung auf Standard-Hardware ohne teure GPU-Ressourcen.',
            why16Structure: 'Musikalische Struktur:',
            why16StructureDesc: '16 Sekunden erfassen typischerweise 1-2 musikalische Phrasen oder Takte (je nach Tempo), was genug Kontext für kreative Fusion ist.',
            why16Focus: 'Fokus:',
            why16FocusDesc: 'Es zwingt dich, den interessantesten Teil jedes Tracks zu wählen, was zu kreativeren Ergebnissen führt!',
            tips: 'Tipps für beste Ergebnisse',
            tipRhythm: 'Komplementäre Rhythmen',
            tipRhythmDesc: 'Wähle einen Track mit starkem Beat und einen anderen mit überzeugender Melodie. Dies erzeugt interessantes Zusammenspiel zwischen Rhythmus und melodischen Elementen.',
            tipHarmony: 'Harmonische Kompatibilität',
            tipHarmonyDesc: 'Tracks in derselben oder verwandten Tonart (z.B. C-Dur und A-Moll) funktionieren am besten zusammen. Dies stellt sicher, dass Harmonien sanft verschmelzen ohne zu kollidieren.',
            tipEnergy: 'Energie-Balance',
            tipEnergyDesc: 'Mische energiereiche Abschnitte mit ruhigeren für interessanten dynamischen Kontrast. Dies fügt emotionale Tiefe hinzu und hält Zuhörer engagiert.',
            tipTimbral: 'Klangfarben-Vielfalt',
            tipTimbralDesc: 'Verschiedene Instrumente (z.B. Gitarre + Klavier) erzeugen reichhaltigere Fusionen. Die Kombination verschiedener Klangfarben fügt Textur und Komplexität hinzu.',
            tipSilence: 'Stille vermeiden',
            tipSilenceDesc: 'Wähle Segmente mit tatsächlichem musikalischem Inhalt, nicht stille oder sehr leise Teile. Die KI benötigt ausreichend Informationen, um überzeugende Muster zu generieren.',
            tipExperiment: 'Experimentiere!',
            tipExperimentDesc: 'Probiere unerwartete Kombinationen - manchmal kommen die kreativsten Ergebnisse von unwahrscheinlichen Paarungen. Hab keine Angst, verschiedene Genres zu mischen!',
            technical: 'Technische Details',
            needHelp: 'Brauchst du Hilfe?',
            needHelpDesc: 'Klicke auf das schwebende Achtelnoten-Symbol in der unteren rechten Ecke, um mit MusicNote, unserem KI-Musikassistenten, zu chatten! MusicNote kann dir helfen mit:',
            helpUnderstand: 'Verständnis, wie das KI-Modell von MusicLab funktioniert',
            helpChoose: 'Auswahl der besten Musterkombinationen für kreative Ergebnisse',
            helpTheory: 'Musiktheorie-Konzepte und Produktionstechniken',
            helpTechnical: 'Technische Fragen zu Audioformaten und -verarbeitung',
            helpInspiration: 'Kreative Inspiration und Vorschläge für deine nächste Fusion',
            helpTrouble: 'Fehlerbehebung bei auftretenden Problemen',
            helpConversation: 'MusicNote behält den Gesprächsverlauf während deiner Sitzung bei, sodass du einen natürlichen Dialog über deinen Musikschaffungsprozess führen kannst. Der Chatbot ist sachkundig, freundlich und immer bereit, dir zu helfen, die kreativste Musik zu erstellen!'
        },
        examples: { title: 'Beispiel-Ausgaben', close: 'Schließen' }
    },
    ru: {
        nav: { contact: 'Контакт', examples: 'Примеры', about: 'О проекте' },
        hero: { title: 'Создавайте уникальную музыку с ИИ', subtitle: 'Загрузите два трека и позвольте ИИ смешать их во что-то новое' },
        track: { 
            title1: 'Входной трек A', 
            title2: 'Входной трек B',
            noFile: 'Файл не загружен',
            uploadText: 'Перетащите аудиофайл сюда или нажмите для загрузки',
            uploadHint: 'MP3, WAV, OGG • Мин. 16 секунд'
        },
        upload: { 
            title: 'Загрузите ваши треки', 
            drop: 'Перетащите аудиофайл сюда или нажмите для выбора', 
            browse: 'Выбрать файлы', 
            supported: 'Поддерживается: MP3, WAV, M4A, OGG',
            instruction: 'Загрузите два трека для начала',
            instructionDetail: 'Выберите аудиофайлы длиной 16+ секунд (MP3, WAV, OGG) для трека A и трека B. Кнопка генерации появится автоматически.'
        },
        player: { 
            selection: 'Выбор времени', 
            start: 'Начало', 
            end: 'Конец', 
            seconds: 'секунд', 
            preview: 'Предпросмотр отрывка', 
            stop: 'Стоп', 
            play: 'Воспроизвести весь трек', 
            volume: 'Громкость',
            playSelection: 'Воспроизвести выбранное',
            clearTrack: 'Очистить трек'
        },
        generate: { button: 'Сгенерировать новый трек', processing: 'Обработка...', ready: 'Загрузите оба трека для генерации' },
        result: { title: 'Сгенерированный результат', play: 'Воспроизвести результат', stop: 'Стоп', download: 'Скачать', downloadTrack: 'Скачать трек' },
        contact: { label: 'Контактный email:' },
        about: { 
            title: 'О MusicLab',
            welcome: 'Добро пожаловать в MusicLab',
            whatIs: 'Что такое MusicLab?',
            whatIsDesc: 'MusicLab — это инновационная платформа для генерации музыки на основе ИИ, которая создает совершенно новую музыку, объединяя паттерны из двух входных треков. Думайте о ней как о креативном музыкальном миксере, который учится на ваших любимых песнях и генерирует что-то уникальное!',
            howWorks: 'Как это работает?',
            step1Title: 'Загрузите ваши треки',
            step1Desc: 'Выберите два аудиофайла (MP3, WAV, M4A или FLAC) - Трек A и Трек B. Это могут быть любые жанры или стили!',
            step2Title: 'Выберите временные окна',
            step2Desc: 'Используйте ползунки, чтобы выбрать 16-секундный сегмент из каждого трека. Здесь вы выбираете паттерны, которые хотите объединить.',
            step3Title: 'Магия ИИ в действии',
            step3Desc: 'Наша модель SimpleTransformer (16.7M параметров) анализирует оба сегмента с использованием аудиокодирования EnCodec и генерирует новые музыкальные паттерны.',
            step4Title: 'Скачайте ваше творение',
            step4Desc: 'Прослушайте сгенерированную музыку и скачайте её как WAV-файл!',
            close: 'Закрыть',
            interface: 'Понимание интерфейса',
            interfaceStep1: 'Шаг 1: Загрузите вашу музыку',
            interfaceStep2: 'Шаг 2: Выберите ваши паттерны',
            interfaceStep3: 'Шаг 3: Сгенерируйте и скачайте',
            dragUpload: 'Перетаскивание/Загрузка:',
            dragUploadDesc: 'Вы можете перетащить аудиофайлы прямо в области загрузки или нажать, чтобы выбрать на компьютере. Каждая область трека принимает файлы MP3, WAV, OGG или FLAC с минимальной длительностью 16 секунд.',
            trackAB: 'Трек A и Трек B:',
            trackABDesc: 'Загрузите два разных музыкальных трека, которые вы хотите объединить. ИИ изучит паттерны обоих треков, чтобы создать что-то совершенно новое.',
            chatbot: 'Музыкальный чат-бот:',
            chatbotDesc: 'Заметили плавающий значок восьмой ноты в правом нижнем углу? Это MusicNote, ваш музыкальный помощник на базе ИИ! Нажмите в любое время, чтобы получить помощь с выбором паттернов, советы по теории музыки или ответы на вопросы о работе MusicLab.',
            waveform: 'Отображение волны:',
            waveformDesc: 'После загрузки вы увидите визуальное представление волны всего трека. Это поможет вам определить интересные секции с сильными битами, мелодиями или текстурами.',
            slidingWindow: 'Управление ползунком:',
            slidingWindowDesc: 'Используйте ползунок под каждой волной, чтобы выбрать именно тот 16-секундный сегмент, который вы хотите использовать. Синяя выделенная область на волне показывает ваш текущий выбор. Вы можете видеть время начала и окончания (в секундах) под волной.',
            playEntire: 'Воспроизвести весь паттерн:',
            playEntireDesc: 'Нажмите большую кнопку воспроизведения под волной, чтобы прослушать ваш полностью загруженный трек от начала до конца. Это поможет вам изучить всю песню перед выбором.',
            previewSel: 'Предпросмотр выбора:',
            previewSelDesc: 'Используйте кнопку "Предпросмотр выбора", чтобы прослушать только 16-секундный сегмент, который вы выбрали ползунком. Это именно та часть, которая будет отправлена ИИ для генерации музыки.',
            clearTrackTitle: 'Очистить трек:',
            clearTrackDesc: 'Если вы хотите начать заново, нажмите "Очистить трек", чтобы удалить загруженный файл и начать снова.',
            durationInd: 'Индикатор длительности:',
            durationIndDesc: 'Синий индикатор "16.0 sec" подтверждает, что вы выбрали действительный 16-секундный паттерн. Оба трека должны иметь выбор, прежде чем вы сможете генерировать музыку.',
            playGen: 'Воспроизвести сгенерированную музыку:',
            playGenDesc: 'После завершения генерации вы увидите новую волну, показывающую вашу созданную ИИ музыку. Нажмите кнопку воспроизведения, чтобы прослушать результат. Плеер показывает текущее время воспроизведения и позволяет регулировать громкость.',
            downloadTitle: 'Скачать трек:',
            downloadDesc: 'Нравится то, что вы создали? Нажмите кнопку "Скачать трек", чтобы сохранить вашу сгенерированную музыку как высококачественный WAV-файл на ваш компьютер. Вы можете использовать её в своих проектах!',
            createAnother: 'Создать ещё:',
            createAnotherDesc: 'Хотите попробовать разные паттерны или треки? Нажмите "Создать ещё", чтобы вернуться к началу и начать новый эксперимент по объединению.',
            assistantTitle: 'Помощник MusicNote:',
            assistantDesc: 'Чат-бот всегда доступен для помощи! Вы можете задавать вопросы типа "Какие паттерны хорошо работают вместе?" или "Как я могу улучшить мои результаты?" MusicNote обладает глубокими знаниями теории музыки и может направить вас к лучшим творческим комбинациям.',
            why16: 'Почему 16 секунд?',
            why16Desc: 'Мы используем 16-секундные сегменты по нескольким техническим и творческим причинам:',
            why16Pattern: 'Распознавание паттернов:',
            why16PatternDesc: '16 секунд достаточно долго для ИИ, чтобы распознать значимые музыкальные паттерны, такие как ритм, мелодия и гармония.',
            why16Efficiency: 'Эффективность модели:',
            why16EfficiencyDesc: 'Наш кодировщик EnCodec работает оптимально с этой длительностью, балансируя качество и скорость обработки.',
            why16Memory: 'Ограничения памяти:',
            why16MemoryDesc: 'Более короткие сегменты позволяют эффективную обработку на стандартном оборудовании без необходимости дорогих GPU-ресурсов.',
            why16Structure: 'Музыкальная структура:',
            why16StructureDesc: '16 секунд обычно захватывают 1-2 музыкальные фразы или такта (в зависимости от темпа), что является достаточным контекстом для творческого объединения.',
            why16Focus: 'Фокус:',
            why16FocusDesc: 'Это заставляет вас выбрать самую интересную часть каждого трека, что приводит к более творческим результатам!',
            tips: 'Советы для лучших результатов',
            tipRhythm: 'Дополняющие ритмы',
            tipRhythmDesc: 'Выберите один трек с сильным битом и другой с убедительной мелодией. Это создает интересное взаимодействие между ритмом и мелодическими элементами.',
            tipHarmony: 'Гармоническая совместимость',
            tipHarmonyDesc: 'Треки в одной или родственных тональностях (например, C мажор и A минор) работают лучше всего вместе. Это гарантирует, что гармонии сливаются плавно без столкновений.',
            tipEnergy: 'Баланс энергии',
            tipEnergyDesc: 'Смешивайте высокоэнергетичные секции с более спокойными для интересного динамического контраста. Это добавляет эмоциональную глубину и удерживает слушателей.',
            tipTimbral: 'Тембральное разнообразие',
            tipTimbralDesc: 'Разные инструменты (например, гитара + фортепиано) создают более богатые объединения. Сочетание различных тембров добавляет текстуру и сложность.',
            tipSilence: 'Избегайте тишины',
            tipSilenceDesc: 'Выбирайте сегменты с фактическим музыкальным содержанием, а не тихие или очень тихие части. ИИ нужна достаточная информация для генерации убедительных паттернов.',
            tipExperiment: 'Экспериментируйте!',
            tipExperimentDesc: 'Попробуйте неожиданные комбинации - иногда самые творческие результаты приходят из неожиданных пар. Не бойтесь смешивать разные жанры!',
            technical: 'Технические детали',
            needHelp: 'Нужна помощь?',
            needHelpDesc: 'Нажмите на плавающий значок восьмой ноты в правом нижнем углу, чтобы пообщаться с MusicNote, нашим музыкальным помощником на базе ИИ! MusicNote может помочь вам с:',
            helpUnderstand: 'Пониманием работы ИИ-модели MusicLab',
            helpChoose: 'Выбором лучших комбинаций паттернов для творческих результатов',
            helpTheory: 'Концепциями теории музыки и техниками производства',
            helpTechnical: 'Техническими вопросами об аудиоформатах и обработке',
            helpInspiration: 'Творческим вдохновением и предложениями для вашего следующего объединения',
            helpTrouble: 'Устранением неполадок, с которыми вы сталкиваетесь',
            helpConversation: 'MusicNote сохраняет историю разговора на протяжении всей вашей сессии, так что вы можете вести естественный диалог о вашем процессе создания музыки. Чат-бот знающий, дружелюбный и всегда готов помочь вам создать самую креативную музыку!'
        },
        examples: { title: 'Примеры результатов', close: 'Закрыть' }
    }
};

let currentLanguage = localStorage.getItem('language') || 'en';
let loadedTranslations = {};

async function loadTranslations(lang) {
    // Try to load from JSON file, fallback to embedded translations
    try {
        const response = await fetch(`/locales/${lang}.json`);
        if (response.ok) {
            loadedTranslations[lang] = await response.json();
            return loadedTranslations[lang];
        }
    } catch (e) {
        console.warn(`Failed to load ${lang}.json, using embedded translations`);
    }
    // Fallback to embedded translations if JSON file not found
    return translations[lang] || translations['en'];
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
    
    // Update all elements with data-i18n-html attribute (for HTML content like chatbot welcome)
    document.querySelectorAll('[data-i18n-html]').forEach(element => {
        const key = element.getAttribute('data-i18n-html');
        const keys = key.split('.');
        let value = langData;
        
        for (const k of keys) {
            value = value[k];
            if (!value) break;
        }
        
        if (value) {
            element.innerHTML = value;
            console.log(`[Language] Updated ${key} to: ${value.substring(0, 50)}...`);
        }
    });
    
    // Update language button
    const langMap = { 
        en: 'EN', de: 'DE', ru: 'RU', fr: 'FR', 
        es: 'ES', pt: 'PT', ar: 'AR', zh: '中文' 
    };
    document.getElementById('current-lang').textContent = langMap[lang];
}

class MusicLab {
    constructor() {
        this.tracks = {
            1: { audio: null, buffer: null, source: null, gainNode: null, duration: 0, isPlaying: false, file: null },
            2: { audio: null, buffer: null, source: null, gainNode: null, duration: 0, isPlaying: false, file: null }
        };
        
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.resultAudio = null;
        this.resultIsPlaying = false;
        this.resultBlob = null;
        this.jobId = null;
        
        // Try to restore last job ID from localStorage
        const savedJobId = localStorage.getItem('lastJobId');
        if (savedJobId) {
            console.log(`[DEBUG] Found saved job ID: ${savedJobId}`);
            this.jobId = savedJobId;
        }
        
        this.init();
    }
    
    init() {
        this.setupFileUploads();
        this.setupDragAndDrop();
        this.checkGenerateReady();
    }
    
    // ========================================
    // File Upload Setup
    // ========================================
    setupFileUploads() {
        [1, 2].forEach(trackNum => {
            const fileInput = document.getElementById(`file-input-${trackNum}`);
            const clearBtn = document.getElementById(`clear-btn-${trackNum}`);
            
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e, trackNum));
            clearBtn.addEventListener('click', () => this.clearTrack(trackNum));
        });
    }
    
    setupDragAndDrop() {
        [1, 2].forEach(trackNum => {
            const uploadArea = document.getElementById(`upload-area-${trackNum}`);
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, this.preventDefaults, false);
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => {
                    uploadArea.classList.add('dragover');
                });
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => {
                    uploadArea.classList.remove('dragover');
                });
            });
            
            uploadArea.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.loadAudioFile(files[0], trackNum);
                }
            });
        });
    }
    
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // ========================================
    // File Handling
    // ========================================
    handleFileSelect(event, trackNum) {
        const file = event.target.files[0];
        if (file) {
            this.loadAudioFile(file, trackNum);
        }
    }
    
    async loadAudioFile(file, trackNum) {
        // Check file type
        if (!file.type.startsWith('audio/')) {
            alert('Please upload an audio file (MP3, WAV, OGG, etc.)');
            return;
        }
        
        // Update status
        this.updateStatus(trackNum, 'uploading', 'Loading...');
        
        try {
            // Read file
            const arrayBuffer = await file.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
            
            // Check duration (minimum 16 seconds)
            if (audioBuffer.duration < 16) {
                alert('Audio file must be at least 16 seconds long');
                this.updateStatus(trackNum, 'error', 'File too short');
                return;
            }
            
            // Store track data
            this.tracks[trackNum].buffer = audioBuffer;
            this.tracks[trackNum].duration = audioBuffer.duration;
            this.tracks[trackNum].file = file;
            
            // DEBUG: Check raw audio data
            const rawData = audioBuffer.getChannelData(0);
            const sampleMax = Math.max(...rawData.slice(0, 10000));
            const sampleMin = Math.min(...rawData.slice(0, 10000));
            console.log(`[RAW AUDIO] Track ${trackNum} - First 10k samples: min=${sampleMin.toFixed(4)}, max=${sampleMax.toFixed(4)}, length=${rawData.length}`);
            
            // Show player
            document.getElementById(`upload-area-${trackNum}`).style.display = 'none';
            document.getElementById(`player-${trackNum}`).style.display = 'block';
            
            // Draw waveform
            this.drawWaveform(trackNum, audioBuffer);
            
            // Setup player controls
            this.setupPlayerControls(trackNum);
            
            // Setup selection controls
            this.setupSelectionControls(trackNum);
            
            // Update status
            this.updateStatus(trackNum, 'ready', `${file.name} (${this.formatTime(audioBuffer.duration)})`);
            
            // Check if can generate
            this.checkGenerateReady();
            
        } catch (error) {
            console.error('Error loading audio:', error);
            alert('Error loading audio file. Please try another file.');
            this.updateStatus(trackNum, 'error', 'Load failed');
        }
    }
    
    // ========================================
    // Waveform Visualization
    // ========================================
    drawWaveform(trackNum, audioBuffer) {
        const canvas = document.getElementById(`waveform-${trackNum}`);
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = canvas.offsetHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        
        // Get audio data
        const data = audioBuffer.getChannelData(0);
        const step = Math.ceil(data.length / width);
        const amp = height / 2;
        
        // Find max amplitude for normalization
        let maxAmp = 0;
        for (let i = 0; i < data.length; i++) {
            maxAmp = Math.max(maxAmp, Math.abs(data[i]));
        }
        
        // Scale to use 95% of canvas height
        const scale = maxAmp > 0 ? 0.95 / maxAmp : 1.0;
        console.log(`[WAVEFORM] Track ${trackNum} - maxAmp=${maxAmp.toFixed(4)}, scale=${scale.toFixed(2)}, canvasHeight=${height}px`);
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw waveform
        ctx.beginPath();
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        
        // Sample a few points for debugging
        const samplePoints = [0, Math.floor(width/4), Math.floor(width/2), Math.floor(3*width/4)];
        const debugSamples = [];
        
        for (let i = 0; i < width; i++) {
            const slice = data.slice(i * step, (i + 1) * step);
            const min = Math.min(...slice);
            const max = Math.max(...slice);
            
            // Log samples for debugging
            if (samplePoints.includes(i)) {
                debugSamples.push({x: i, rawMin: min.toFixed(4), rawMax: max.toFixed(4)});
            }
            
            // Apply scaling and convert to canvas coordinates
            const minScaled = min * scale;
            const maxScaled = max * scale;
            const minY = amp - (minScaled * amp);
            const maxY = amp - (maxScaled * amp);
            
            if (i === 0) {
                ctx.moveTo(i, minY);
            }
            ctx.lineTo(i, maxY);
            ctx.lineTo(i, minY);
        }
        
        console.log(`[WAVEFORM SAMPLES] Track ${trackNum} - Sample points:`, debugSamples);
        
        ctx.stroke();
        
        // Draw time markers
        this.drawTimeMarkers(trackNum, audioBuffer.duration);
        
        // Restore current selection overlay (don't reset to 0-16)
        const startTime = parseFloat(document.getElementById(`start-time-${trackNum}`).value) || 0;
        const endTime = parseFloat(document.getElementById(`end-time-${trackNum}`).value) || 16;
        this.updateSelectionOverlay(trackNum, startTime, endTime);
    }
    
    drawTimeMarkers(trackNum, duration) {
        const container = document.getElementById(`markers-${trackNum}`);
        container.innerHTML = '';
        
        const numMarkers = 5;
        for (let i = 0; i <= numMarkers; i++) {
            const time = (duration / numMarkers) * i;
            const marker = document.createElement('span');
            marker.textContent = this.formatTime(time);
            marker.style.flex = '1';
            marker.style.textAlign = i === 0 ? 'left' : i === numMarkers ? 'right' : 'center';
            container.appendChild(marker);
        }
    }
    
    updateSelectionOverlay(trackNum, startTime, endTime) {
        const overlay = document.getElementById(`selection-${trackNum}`);
        const duration = this.tracks[trackNum].duration;
        
        const startPercent = (startTime / duration) * 100;
        const widthPercent = ((endTime - startTime) / duration) * 100;
        
        console.log(`[OVERLAY] Track ${trackNum} - Setting overlay: left=${startPercent.toFixed(2)}%, width=${widthPercent.toFixed(2)}% (start=${startTime.toFixed(2)}s, end=${endTime.toFixed(2)}s, duration=${duration.toFixed(2)}s)`);
        
        overlay.style.left = `${startPercent}%`;
        overlay.style.width = `${widthPercent}%`;
    }

    updateWaveformProgress(trackNum, progress) {
        const canvas = document.getElementById(`waveform-${trackNum}`);
        if (!canvas) return;

        const track = this.tracks[trackNum];
        if (!track.buffer) return;

        const ctx = canvas.getContext('2d');
        
        // Set canvas size (same as drawWaveform)
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = canvas.offsetHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        
        // Get audio data
        const data = track.buffer.getChannelData(0);
        const step = Math.ceil(data.length / width);
        const amp = height / 2;
        
        // Find max amplitude for normalization
        let maxAmp = 0;
        for (let i = 0; i < data.length; i++) {
            maxAmp = Math.max(maxAmp, Math.abs(data[i]));
        }
        const scale = maxAmp > 0 ? 0.95 / maxAmp : 1; // Scale to 95% of available height
        
        // Get time window info
        const startTime = track.startTimeOffset || 0;
        const duration = track.duration;
        const startPercent = startTime / duration;
        const playDuration = track.playDuration || duration;
        const endPercent = (startTime + playDuration) / duration;
        
        // Calculate pixel positions for time window
        const windowStartX = Math.floor(width * startPercent);
        const windowWidth = Math.floor(width * (endPercent - startPercent));
        const progressInWindow = Math.floor(windowWidth * progress);
        const progressAbsoluteX = windowStartX + progressInWindow;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Draw base waveform
        ctx.beginPath();
        ctx.strokeStyle = '#6366f1';
        ctx.lineWidth = 2;
        
        for (let i = 0; i < width; i++) {
            const min = Math.min(...data.slice(i * step, (i + 1) * step)) * scale;
            const max = Math.max(...data.slice(i * step, (i + 1) * step)) * scale;
            
            if (i === 0) {
                ctx.moveTo(i, (1 + min) * amp);
            }
            ctx.lineTo(i, (1 + max) * amp);
            ctx.lineTo(i, (1 + min) * amp);
        }
        
        ctx.stroke();
        
        // Draw progress overlay (from window start to current position)
        console.log(`[PROGRESS] Track ${trackNum} - progress=${(progress * 100).toFixed(1)}%, startTime=${startTime.toFixed(2)}s, playDuration=${playDuration.toFixed(2)}s, windowStartX=${windowStartX}, progressAbsoluteX=${progressAbsoluteX}`);
        
        if (progressInWindow > 0) {
            ctx.beginPath();
            ctx.strokeStyle = '#00d4aa';
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.6;
            
            for (let i = windowStartX; i < progressAbsoluteX; i++) {
                const min = Math.min(...data.slice(i * step, (i + 1) * step)) * scale;
                const max = Math.max(...data.slice(i * step, (i + 1) * step)) * scale;
                
                if (i === windowStartX) {
                    ctx.moveTo(i, (1 + min) * amp);
                }
                ctx.lineTo(i, (1 + max) * amp);
                ctx.lineTo(i, (1 + min) * amp);
            }
            
            ctx.stroke();
            ctx.globalAlpha = 1.0;
            
            // Draw progress line
            ctx.strokeStyle = '#00d4aa';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(progressAbsoluteX, 0);
            ctx.lineTo(progressAbsoluteX, height);
            ctx.stroke();
        }
        
        // Redraw selection overlay
        const selectionStart = parseFloat(document.getElementById(`start-time-${trackNum}`).value);
        const selectionEnd = parseFloat(document.getElementById(`end-time-${trackNum}`).value);
        console.log(`[REDRAW] Track ${trackNum} - Redrawing overlay after progress update: selectionStart=${selectionStart.toFixed(2)}s, selectionEnd=${selectionEnd.toFixed(2)}s`);
        this.updateSelectionOverlay(trackNum, selectionStart, selectionEnd);
    }
    
    // ========================================
    // Player Controls
    // ========================================
    setupPlayerControls(trackNum) {
        // Only setup listeners once per track to prevent duplicates
        if (!this.tracks[trackNum].listenersSetup) {
            const playBtn = document.getElementById(`play-btn-${trackNum}`);
            const volumeSlider = document.getElementById(`volume-${trackNum}`);
            const previewBtn = document.getElementById(`preview-btn-${trackNum}`);
            
            // Clone and replace to remove any old listeners
            const playBtnNew = playBtn.cloneNode(true);
            playBtn.parentNode.replaceChild(playBtnNew, playBtn);
            
            const volumeSliderNew = volumeSlider.cloneNode(true);
            volumeSlider.parentNode.replaceChild(volumeSliderNew, volumeSlider);
            
            const previewBtnNew = previewBtn.cloneNode(true);
            previewBtn.parentNode.replaceChild(previewBtnNew, previewBtn);
            
            // Add event listeners to new nodes
            document.getElementById(`play-btn-${trackNum}`).addEventListener('click', () => this.togglePlay(trackNum));
            document.getElementById(`volume-${trackNum}`).addEventListener('input', (e) => this.setVolume(trackNum, e.target.value));
            document.getElementById(`preview-btn-${trackNum}`).addEventListener('click', () => this.previewSelection(trackNum));
            
            this.tracks[trackNum].listenersSetup = true;
            console.log(`[DEBUG] Track ${trackNum} player controls setup`);
        }
        
        // Initialize duration display
        document.getElementById(`duration-${trackNum}`).textContent = 
            this.formatTime(this.tracks[trackNum].duration);
    }
    
    togglePlay(trackNum) {
        if (this.tracks[trackNum].isPlaying) {
            this.stopTrack(trackNum);
        } else {
            // Play full track from beginning
            this.playTrack(trackNum, 0, null);
        }
    }
    
    async playTrack(trackNum, startTime = 0, endTime = null) {
        // Resume AudioContext if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        
        const track = this.tracks[trackNum];
        
        // Stop if already playing
        if (track.source) {
            try {
                track.source.stop();
                track.source.disconnect();
            } catch (e) {}
        }
        
        // Calculate duration
        const duration = endTime ? endTime - startTime : track.duration - startTime;
        console.log(`[PLAY] Track ${trackNum} - startTime=${startTime.toFixed(2)}s, endTime=${endTime ? endTime.toFixed(2) : 'null'}, duration=${duration.toFixed(2)}s`);
        
        // Create source
        track.source = this.audioContext.createBufferSource();
        track.source.buffer = track.buffer;
        
        // Create gain node for volume
        track.gainNode = this.audioContext.createGain();
        const volume = document.getElementById(`volume-${trackNum}`).value / 100;
        track.gainNode.gain.value = volume;
        
        console.log(`[VOLUME] Track ${trackNum} - Set volume to ${(volume * 100).toFixed(0)}%`);
        
        // Connect
        track.source.connect(track.gainNode);
        track.gainNode.connect(this.audioContext.destination);
        
        // Play
        track.source.start(0, startTime, duration);
        track.isPlaying = true;
        
        // Update UI
        this.updatePlayButton(trackNum, true);
        
        // Update time display and waveform progress
        track.startTimestamp = Date.now();
        track.startTimeOffset = startTime;
        track.playDuration = duration;
        
        const updateTime = () => {
            if (!track.isPlaying) return;
            
            const elapsed = (Date.now() - track.startTimestamp) / 1000;
            const currentTime = track.startTimeOffset + Math.min(elapsed, track.playDuration);
            
            const currentTimeEl = document.getElementById(`current-time-${trackNum}`);
            if (currentTimeEl) {
                currentTimeEl.textContent = this.formatTime(currentTime);
            }
            
            // Update waveform progress
            const progress = Math.min(elapsed / track.playDuration, 1);
            this.updateWaveformProgress(trackNum, progress);
            
            track.animationFrame = requestAnimationFrame(updateTime);
        };
        updateTime();
        
        // Handle end
        track.source.onended = () => {
            if (track.isPlaying) {
                this.stopTrack(trackNum);
            }
        };
    }
    
    stopTrack(trackNum) {
        const track = this.tracks[trackNum];
        
        if (track.source) {
            try {
                track.source.stop();
                track.source.disconnect();
            } catch (e) {}
            track.source = null;
        }
        
        if (track.gainNode) {
            track.gainNode.disconnect();
            track.gainNode = null;
        }
        
        console.log(`[STOP] Track ${trackNum} - Playback stopped`);
        
        track.isPlaying = false;
        
        // Cancel animation frame
        if (track.animationFrame) {
            cancelAnimationFrame(track.animationFrame);
            track.animationFrame = null;
        }
        
        this.updatePlayButton(trackNum, false);
        
        const currentTimeEl = document.getElementById(`current-time-${trackNum}`);
        if (currentTimeEl) {
            currentTimeEl.textContent = '0:00';
        }
        
        // Reset waveform to show no progress
        if (track.buffer) {
            this.drawWaveform(trackNum, track.buffer);
        }
    }
    
    updatePlayButton(trackNum, isPlaying) {
        const playBtn = document.getElementById(`play-btn-${trackNum}`);
        const playIcon = playBtn.querySelector('.icon-play');
        const pauseIcon = playBtn.querySelector('.icon-pause');
        
        if (isPlaying) {
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'block';
        } else {
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
        }
    }
    
    setVolume(trackNum, value) {
        const track = this.tracks[trackNum];
        if (track.gainNode) {
            track.gainNode.gain.value = value / 100;
            console.log(`[VOLUME] Track ${trackNum} - Volume changed to ${value}%`);
        }
    }
    
    previewSelection(trackNum) {
        const startTime = parseFloat(document.getElementById(`start-time-${trackNum}`).value);
        const endTime = parseFloat(document.getElementById(`end-time-${trackNum}`).value);
        
        console.log(`[PREVIEW] Track ${trackNum} - Preview Selection clicked: startTime=${startTime.toFixed(2)}s, endTime=${endTime.toFixed(2)}s`);
        
        this.stopTrack(trackNum);
        this.playTrack(trackNum, startTime, endTime);
    }
    
    // ========================================
    // Selection Controls
    // ========================================
    setupSelectionControls(trackNum) {
        const slider = document.getElementById(`slider-${trackNum}`);
        const startInput = document.getElementById(`start-time-${trackNum}`);
        const endInput = document.getElementById(`end-time-${trackNum}`);
        const durationDisplay = document.getElementById(`selection-duration-${trackNum}`);
        
        const duration = this.tracks[trackNum].duration;
        
        // Set slider max
        slider.max = duration - 16;
        slider.value = 0;
        
        // Set input constraints
        startInput.max = duration - 16;
        endInput.max = duration;
        endInput.min = 16;
        
        // Slider change
        slider.addEventListener('input', (e) => {
            const startTime = parseFloat(e.target.value);
            const endTime = startTime + 16;
            
            console.log(`[SLIDER] Track ${trackNum} - slider moved to startTime=${startTime.toFixed(2)}s, endTime=${endTime.toFixed(2)}s`);
            
            startInput.value = startTime.toFixed(1);
            endInput.value = endTime.toFixed(1);
            
            this.updateSelectionOverlay(trackNum, startTime, endTime);
            this.updateSelectionDuration(trackNum);
        });
        
        // Input changes
        startInput.addEventListener('change', () => {
            let startTime = parseFloat(startInput.value);
            const maxStart = duration - 16;
            
            startTime = Math.max(0, Math.min(startTime, maxStart));
            startInput.value = startTime.toFixed(1);
            
            const endTime = startTime + 16;
            endInput.value = endTime.toFixed(1);
            
            slider.value = startTime;
            this.updateSelectionOverlay(trackNum, startTime, endTime);
            this.updateSelectionDuration(trackNum);
        });
        
        endInput.addEventListener('change', () => {
            let endTime = parseFloat(endInput.value);
            endTime = Math.max(16, Math.min(endTime, duration));
            
            const startTime = endTime - 16;
            startInput.value = startTime.toFixed(1);
            endInput.value = endTime.toFixed(1);
            
            slider.value = startTime;
            this.updateSelectionOverlay(trackNum, startTime, endTime);
            this.updateSelectionDuration(trackNum);
        });
    }
    
    updateSelectionDuration(trackNum) {
        const startTime = parseFloat(document.getElementById(`start-time-${trackNum}`).value);
        const endTime = parseFloat(document.getElementById(`end-time-${trackNum}`).value);
        const duration = endTime - startTime;
        
        document.getElementById(`selection-duration-${trackNum}`).textContent = 
            `${duration.toFixed(1)} sec`;
    }
    
    // ========================================
    // Status Updates
    // ========================================
    updateStatus(trackNum, state, text) {
        const statusElement = document.getElementById(`status-track${trackNum}`);
        const indicator = statusElement.querySelector('.status-indicator');
        const statusText = statusElement.querySelector('.status-text');
        
        indicator.className = 'status-indicator';
        if (state === 'uploading') {
            indicator.classList.add('uploading');
        } else if (state === 'ready') {
            indicator.classList.add('ready');
        }
        
        statusText.textContent = text;
    }
    
    // ========================================
    // Clear Track
    // ========================================
    clearTrack(trackNum) {
        // Stop playback
        this.stopTrack(trackNum);
        
        // Clear data
        this.tracks[trackNum] = {
            audio: null,
            buffer: null,
            source: null,
            duration: 0,
            isPlaying: false,
            file: null
        };
        
        // Reset UI
        document.getElementById(`upload-area-${trackNum}`).style.display = 'block';
        document.getElementById(`player-${trackNum}`).style.display = 'none';
        document.getElementById(`file-input-${trackNum}`).value = '';
        
        this.updateStatus(trackNum, '', 'No file uploaded');
        this.checkGenerateReady();
    }
    
    // ========================================
    // Generate Music
    // ========================================
    checkGenerateReady() {
        const track1Ready = this.tracks[1].buffer !== null;
        const track2Ready = this.tracks[2].buffer !== null;
        
        console.log('Check Generate Ready:', { track1Ready, track2Ready });
        
        const generateSection = document.getElementById('generate-section');
        const statusMessage = document.getElementById('status-message');
        
        if (track1Ready && track2Ready) {
            console.log('Both tracks ready - showing Generate button');
            generateSection.style.display = 'block';
            if (statusMessage) statusMessage.style.display = 'none';
            this.setupGenerateButton();
        } else {
            generateSection.style.display = 'none';
            if (statusMessage) statusMessage.style.display = 'block';
        }
    }
    
    setupGenerateButton() {
        const generateBtn = document.getElementById('generate-btn');
        
        // Remove old listeners
        const newBtn = generateBtn.cloneNode(true);
        generateBtn.parentNode.replaceChild(newBtn, generateBtn);
        
        newBtn.addEventListener('click', () => this.generateMusic());
    }
    
    async generateMusic() {
        // Get selections
        const track1Start = parseFloat(document.getElementById('start-time-1').value);
        const track1End = parseFloat(document.getElementById('end-time-1').value);
        const track2Start = parseFloat(document.getElementById('start-time-2').value);
        const track2End = parseFloat(document.getElementById('end-time-2').value);
        
        // Show loading
        document.getElementById('generate-section').style.display = 'none';
        document.getElementById('loading-section').style.display = 'block';
        
        try {
            // Upload files and start generation
            await this.callGenerationAPI(track1Start, track1End, track2Start, track2End);
            
            // Show result
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('result-section').style.display = 'block';
            
            this.setupResultPlayer();
        } catch (error) {
            console.error('Generation failed:', error);
            alert('Failed to generate music. Please try again.');
            document.getElementById('loading-section').style.display = 'none';
            document.getElementById('generate-section').style.display = 'block';
        }
    }
    
    async callGenerationAPI(track1Start, track1End, track2Start, track2End) {
        const API_URL = window.location.hostname === 'localhost' ? 'http://localhost:8001' : `http://${window.location.hostname}:8001`;
        
        // Create FormData with files and parameters
        const formData = new FormData();
        formData.append('track1', this.tracks[1].file);
        formData.append('track2', this.tracks[2].file);
        formData.append('start_time_1', track1Start.toString());
        formData.append('end_time_1', track1End.toString());
        formData.append('start_time_2', track2Start.toString());
        formData.append('end_time_2', track2End.toString());
        
        // Submit generation request
        const response = await fetch(`${API_URL}/api/generate`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        this.jobId = data.job_id;
        
        // Save job ID to localStorage for persistence across page refreshes
        localStorage.setItem('lastJobId', this.jobId);
        console.log(`[DEBUG] Job started with ID: ${this.jobId}`);
        
        // Poll for status
        await this.pollGenerationStatus();
    }
    
    async pollGenerationStatus() {
        const API_URL = window.location.hostname === 'localhost' ? 'http://localhost:8001' : `http://${window.location.hostname}:8001`;
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const loadingSubtitle = document.querySelector('.loading-subtitle');
        
        const startTime = Date.now();
        const timeout = 120000; // 2 minutes timeout
        
        while (true) {
            const elapsed = Date.now() - startTime;
            if (elapsed > timeout) {
                throw new Error('Generation timeout (2 minutes). Please try again with shorter audio files.');
            }
            
            try {
                const response = await fetch(`${API_URL}/api/status/${this.jobId}`, {
                    timeout: 5000
                });
                
                if (!response.ok) {
                    throw new Error(`Status check failed: ${response.status}`);
                }
                
                const status = await response.json();
                
                console.log('Generation status:', status);
                
                // Update progress bar
                progressFill.style.width = `${status.progress}%`;
                progressText.textContent = `${status.progress}%`;
                
                // Update status message
                if (loadingSubtitle && status.message) {
                    loadingSubtitle.textContent = status.message;
                }
                
                if (status.status === 'completed') {
                    this.outputPath = status.output_path;
                    console.log('Generation completed!', this.outputPath);
                    break;
                } else if (status.status === 'failed') {
                    throw new Error(status.message || 'Generation failed');
                }
                
                // Wait 1 second before next poll
                await new Promise(resolve => setTimeout(resolve, 1000));
            } catch (error) {
                console.error('Poll error:', error);
                // If it's a timeout error, throw it
                if (error.message.includes('timeout')) {
                    throw error;
                }
                // For other errors, wait and retry
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }
    
    // ========================================
    // Result Player
    // ========================================
    async setupResultPlayer() {
        const playBtn = document.getElementById('play-btn-result');
        const downloadBtn = document.getElementById('download-btn');
        const newGenBtn = document.getElementById('new-generation-btn');
        
        // Remove old event listeners by cloning and replacing nodes
        // This prevents duplicate listeners on regeneration
        if (!this.resultListenersSetup) {
            const playBtnNew = playBtn.cloneNode(true);
            playBtn.parentNode.replaceChild(playBtnNew, playBtn);
            
            const downloadBtnNew = downloadBtn.cloneNode(true);
            downloadBtn.parentNode.replaceChild(downloadBtnNew, downloadBtn);
            
            const newGenBtnNew = newGenBtn.cloneNode(true);
            newGenBtn.parentNode.replaceChild(newGenBtnNew, newGenBtn);
            
            // Add event listeners to new nodes
            document.getElementById('play-btn-result').addEventListener('click', () => this.toggleResultPlay());
            document.getElementById('download-btn').addEventListener('click', () => this.downloadResult());
            document.getElementById('new-generation-btn').addEventListener('click', () => this.resetForNewGeneration());
            
            this.resultListenersSetup = true;
            console.log('[DEBUG] Result player event listeners setup');
        }
        
        // Load generated audio from API
        await this.loadGeneratedAudio();
        
        document.getElementById('duration-result').textContent = 
            this.formatTime(this.resultAudio.duration);
        
        this.drawResultWaveform();
    }
    
    async loadGeneratedAudio() {
        const API_URL = window.location.hostname === 'localhost' ? 'http://localhost:8001' : `http://${window.location.hostname}:8001`;
        
        try {
            console.log(`[DEBUG] Fetching audio for job: ${this.jobId}`);
            
            // Fetch generated audio file
            const response = await fetch(`${API_URL}/api/download/${this.jobId}`);
            
            console.log(`[DEBUG] Download response status: ${response.status}`);
            console.log(`[DEBUG] Download response headers:`, response.headers.get('content-length'));
            
            if (!response.ok) {
                throw new Error('Failed to download generated audio');
            }
            
            // Convert to audio buffer
            const arrayBuffer = await response.arrayBuffer();
            console.log(`[DEBUG] ArrayBuffer size: ${arrayBuffer.byteLength} bytes`);
            
            // IMPORTANT: Create blob BEFORE decoding, as decodeAudioData consumes the ArrayBuffer
            this.resultBlob = new Blob([arrayBuffer], { type: 'audio/wav' });
            console.log(`[DEBUG] Blob created, size: ${this.resultBlob.size} bytes`);
            
            // Clone the arrayBuffer for decoding (since decodeAudioData transfers/consumes it)
            const arrayBufferCopy = arrayBuffer.slice(0);
            this.resultAudio = await this.audioContext.decodeAudioData(arrayBufferCopy);
            console.log(`[DEBUG] Audio decoded successfully, duration: ${this.resultAudio.duration}s`);
            
        } catch (error) {
            console.error('Error loading generated audio:', error);
            // Fallback to track 1 for demo
            this.resultAudio = this.tracks[1].buffer;
        }
    }
    
    drawResultWaveform() {
        const canvas = document.getElementById('waveform-result');
        const ctx = canvas.getContext('2d');
        
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = canvas.offsetHeight * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        
        const data = this.resultAudio.getChannelData(0);
        const step = Math.ceil(data.length / width);
        const amp = height / 2;
        
        ctx.clearRect(0, 0, width, height);
        ctx.beginPath();
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        
        for (let i = 0; i < width; i++) {
            const min = Math.min(...data.slice(i * step, (i + 1) * step));
            const max = Math.max(...data.slice(i * step, (i + 1) * step));
            
            if (i === 0) {
                ctx.moveTo(i, (1 + min) * amp);
            }
            ctx.lineTo(i, (1 + max) * amp);
            ctx.lineTo(i, (1 + min) * amp);
        }
        
        ctx.stroke();
    }
    
    toggleResultPlay() {
        if (this.resultIsPlaying) {
            this.stopResultPlay();
        } else {
            this.playResultAudio();
        }
    }
    
    async playResultAudio() {
        // Resume AudioContext if suspended (required by browser autoplay policy)
        if (this.audioContext.state === 'suspended') {
            console.log('[DEBUG] AudioContext suspended, resuming...');
            await this.audioContext.resume();
        }
        
        // Clean up any existing source
        if (this.resultSource) {
            try {
                this.resultSource.stop();
                console.log('[DEBUG] Stopped existing source');
            } catch (e) {
                console.log('[DEBUG] Could not stop existing source:', e.message);
            }
            this.resultSource.disconnect();
            this.resultSource = null;
        }
        
        console.log('[DEBUG] Playing result audio, buffer:', this.resultAudio);
        console.log('[DEBUG] AudioContext state:', this.audioContext.state);
        
        try {
            const source = this.audioContext.createBufferSource();
            source.buffer = this.resultAudio;
            
            const gainNode = this.audioContext.createGain();
            gainNode.gain.value = document.getElementById('volume-result').value / 100;
            
            source.connect(gainNode);
            gainNode.connect(this.audioContext.destination);
            
            source.start(0);
            console.log('[DEBUG] Audio source started successfully');
            
            this.resultIsPlaying = true;
            this.resultSource = source;
            
            const playBtn = document.getElementById('play-btn-result');
            playBtn.querySelector('.icon-play').style.display = 'none';
            playBtn.querySelector('.icon-pause').style.display = 'block';
            
            source.onended = () => {
                console.log('[DEBUG] Audio playback ended');
                this.stopResultPlay();
            };
            
            // Update time
            const startTime = Date.now();
            const updateTime = () => {
                if (this.resultIsPlaying) {
                    const elapsed = (Date.now() - startTime) / 1000;
                    document.getElementById('current-time-result').textContent = 
                        this.formatTime(Math.min(elapsed, this.resultAudio.duration));
                    requestAnimationFrame(updateTime);
                }
            };
            updateTime();
            
        } catch (error) {
            console.error('[ERROR] Failed to play audio:', error);
            alert('Failed to play audio. Please try again.');
            this.resultIsPlaying = false;
        }
    }
    
    stopResultPlay() {
        if (this.resultSource) {
            try {
                this.resultSource.stop();
                this.resultSource.disconnect();
                console.log('[DEBUG] Stopped and disconnected result source');
            } catch (e) {
                console.log('[DEBUG] Could not stop source:', e.message);
            }
            this.resultSource = null;
        }
        
        this.resultIsPlaying = false;
        
        const playBtn = document.getElementById('play-btn-result');
        playBtn.querySelector('.icon-play').style.display = 'block';
        playBtn.querySelector('.icon-pause').style.display = 'none';
        
        document.getElementById('current-time-result').textContent = '0:00';
    }
    
    downloadResult() {
        console.log(`[DEBUG] Download button clicked`);
        console.log(`[DEBUG] resultBlob exists: ${!!this.resultBlob}`);
        console.log(`[DEBUG] resultBlob size: ${this.resultBlob ? this.resultBlob.size : 'N/A'} bytes`);
        
        if (!this.resultBlob) {
            alert('No audio available to download');
            return;
        }
        
        if (this.resultBlob.size === 0) {
            alert('Generated audio file is empty (0 bytes). Please try generating again.');
            return;
        }
        
        // Create download link
        const url = URL.createObjectURL(this.resultBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `musiclab_generated_${Date.now()}.wav`;
        console.log(`[DEBUG] Creating download link: ${a.download}`);
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log(`[DEBUG] Download triggered successfully`);
    }
    
    resetForNewGeneration() {
        // Stop any playing audio
        this.stopResultPlay();
        
        // Clear previous result data
        if (this.resultSource) {
            try {
                this.resultSource.stop();
            } catch (e) {
                // Already stopped
            }
            this.resultSource = null;
        }
        
        this.resultIsPlaying = false;
        
        // Reset UI
        document.getElementById('result-section').style.display = 'none';
        document.getElementById('generate-section').style.display = 'block';
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        
        console.log('[DEBUG] Reset for new generation - cleared result state');
    }
    
    // ========================================
    // Utility Functions
    // ========================================
    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
}

// ========================================
// MusicNote Chatbot
// ========================================

class MusicChatbot {
    constructor() {
        this.API_URL = window.location.hostname === 'localhost' ? 'http://localhost:8001' : `http://${window.location.hostname}:8001`;
        this.sessionId = this.generateSessionId();
        this.isOpen = false;
        
        this.trigger = document.getElementById('chatbot-trigger');
        this.window = document.getElementById('chatbot-window');
        this.closeBtn = document.getElementById('chatbot-close');
        this.messagesContainer = document.getElementById('chatbot-messages');
        this.input = document.getElementById('chatbot-input');
        this.sendBtn = document.getElementById('chatbot-send');
        
        this.setupEventListeners();
        this.setupBeforeUnload();
        
        console.log('[Chatbot] Initialized with session:', this.sessionId);
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    setupEventListeners() {
        this.trigger.addEventListener('click', () => this.toggle());
        this.closeBtn.addEventListener('click', () => this.close());
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        
        this.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
    }
    
    setupBeforeUnload() {
        // Clear chat history when user closes browser/tab
        window.addEventListener('beforeunload', () => {
            this.clearSession();
        });
    }
    
    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }
    
    open() {
        this.window.classList.add('active');
        this.isOpen = true;
        this.input.focus();
        console.log('[Chatbot] Opened');
    }
    
    close() {
        this.window.classList.remove('active');
        this.isOpen = false;
        console.log('[Chatbot] Closed');
    }
    
    async sendMessage() {
        const message = this.input.value.trim();
        if (!message) return;
        
        // Clear input
        this.input.value = '';
        
        // Add user message to UI
        this.addMessage(message, 'user');
        
        // Show typing indicator
        const typingIndicator = this.showTyping();
        
        try {
            // Send to API
            const formData = new FormData();
            formData.append('session_id', this.sessionId);
            formData.append('message', message);
            formData.append('language', currentLanguage);  // Send user's language preference
            
            const response = await fetch(`${this.API_URL}/api/chat`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeTyping(typingIndicator);
            
            // Add bot response
            if (data.status === 'success') {
                this.addMessage(data.response, 'bot');
                console.log('[Chatbot] Exchange count:', data.history_length);
            } else {
                this.addMessage('Sorry, I encountered an error. Please try again! 🎵', 'bot');
            }
            
        } catch (error) {
            console.error('[Chatbot] Error:', error);
            this.removeTyping(typingIndicator);
            this.addMessage('Oops! Connection error. Make sure the server is running. 🎵', 'bot');
        }
    }
    
    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = content.replace(/\n/g, '<br>');
        
        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    showTyping() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message bot-message';
        typingDiv.id = 'typing-indicator';
        
        const typingContent = document.createElement('div');
        typingContent.className = 'message-content chat-typing';
        typingContent.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        
        typingDiv.appendChild(typingContent);
        this.messagesContainer.appendChild(typingDiv);
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        
        return typingDiv;
    }
    
    removeTyping(typingElement) {
        if (typingElement && typingElement.parentNode) {
            typingElement.parentNode.removeChild(typingElement);
        }
    }
    
    async clearSession() {
        try {
            await fetch(`${this.API_URL}/api/chat/${this.sessionId}`, {
                method: 'DELETE',
                keepalive: true  // Allows request to complete during page unload
            });
            console.log('[Chatbot] Session cleared');
        } catch (error) {
            // Silently fail during page unload - this is expected behavior
            // The browser may cancel the request, which is okay
        }
    }
}

// About Modal Functionality
function initializeAboutModal() {
    const modal = document.getElementById('about-modal');
    const btn = document.getElementById('about-btn');
    const closeBtn = document.querySelector('.modal-close');
    const gotItBtn = document.querySelector('.modal-btn');

    if (!modal || !btn) return;

    // Open modal
    btn.addEventListener('click', (e) => {
        e.preventDefault();
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
    });

    // Close modal functions
    const closeModal = () => {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto'; // Restore scrolling
    };

    if (closeBtn) closeBtn.addEventListener('click', closeModal);
    if (gotItBtn) gotItBtn.addEventListener('click', closeModal);

    // Close when clicking outside the modal content
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            closeModal();
        }
    });

    // Close with Escape key
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            closeModal();
        }
    });
}

// Examples Modal Player Class
class ExamplesPlayer {
    constructor() {
        this.audioContext = null;
        this.examples = {
            input: { buffer: null, source: null, gainNode: null, isPlaying: false },
            target: { buffer: null, source: null, gainNode: null, isPlaying: false },
            output: { buffer: null, source: null, gainNode: null, isPlaying: false }
        };
        this.animationFrames = {};
    }

    async initialize() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        await this.loadAllExamples();
        this.setupControlListeners();
    }

    async loadAllExamples() {
        const examples = [
            { id: 'input', name: 'input' },
            { id: 'target', name: 'target' },
            { id: 'output', name: 'output' }
        ];

        for (const example of examples) {
            try {
                const API_URL = window.location.hostname === 'localhost' ? 'http://localhost:8001' : `http://${window.location.hostname}:8001`;
                const response = await fetch(`${API_URL}/api/examples/${example.name}`);
                const arrayBuffer = await response.arrayBuffer();
                const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
                this.examples[example.id].buffer = audioBuffer;
                
                // Draw waveform
                this.drawWaveform(example.id, audioBuffer);
                
                // Update duration
                const duration = this.formatTime(audioBuffer.duration);
                document.getElementById(`${example.id}-duration`).textContent = duration;
                document.getElementById(`${example.id}-total-time`).textContent = duration;
            } catch (error) {
                console.error(`Failed to load ${example.id}:`, error);
            }
        }
    }

    drawWaveform(exampleId, audioBuffer) {
        const canvas = document.getElementById(`${exampleId}-waveform`);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        const channelData = audioBuffer.getChannelData(0);
        const step = Math.ceil(channelData.length / width);
        const amp = height / 2;
        
        ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        for (let i = 0; i < width; i++) {
            let min = 1.0;
            let max = -1.0;
            for (let j = 0; j < step; j++) {
                const datum = channelData[i * step + j];
                if (datum < min) min = datum;
                if (datum > max) max = datum;
            }
            const yMin = (1 + min) * amp;
            const yMax = (1 + max) * amp;
            
            ctx.fillRect(i, yMin, 1, yMax - yMin);
        }
        ctx.stroke();
    }

    updateWaveformProgress(exampleId, progress) {
        const canvas = document.getElementById(`${exampleId}-waveform`);
        if (!canvas) return;

        const example = this.examples[exampleId];
        if (!example.buffer) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        const channelData = example.buffer.getChannelData(0);
        const step = Math.ceil(channelData.length / width);
        const amp = height / 2;
        
        // Draw base waveform
        ctx.fillStyle = 'rgba(102, 126, 234, 0.3)';
        
        for (let i = 0; i < width; i++) {
            let min = 1.0;
            let max = -1.0;
            for (let j = 0; j < step; j++) {
                const datum = channelData[i * step + j];
                if (datum < min) min = datum;
                if (datum > max) max = datum;
            }
            const yMin = (1 + min) * amp;
            const yMax = (1 + max) * amp;
            
            ctx.fillRect(i, yMin, 1, yMax - yMin);
        }
        
        // Draw progress overlay
        const progressX = Math.floor(width * progress);
        if (progressX > 0) {
            ctx.fillStyle = 'rgba(0, 212, 170, 0.4)';
            
            for (let i = 0; i < progressX; i++) {
                let min = 1.0;
                let max = -1.0;
                for (let j = 0; j < step; j++) {
                    const datum = channelData[i * step + j];
                    if (datum < min) min = datum;
                    if (datum > max) max = datum;
                }
                const yMin = (1 + min) * amp;
                const yMax = (1 + max) * amp;
                
                ctx.fillRect(i, yMin, 1, yMax - yMin);
            }
            
            // Draw progress line
            ctx.strokeStyle = '#00d4aa';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(progressX, 0);
            ctx.lineTo(progressX, height);
            ctx.stroke();
        }
    }

    setupControlListeners() {
        ['input', 'target', 'output'].forEach(id => {
            const playBtn = document.getElementById(`${id}-play-btn`);
            const volumeSlider = document.getElementById(`${id}-volume`);
            
            if (playBtn) {
                playBtn.addEventListener('click', () => this.togglePlayback(id));
            }
            
            if (volumeSlider) {
                volumeSlider.addEventListener('input', (e) => {
                    this.setVolume(id, e.target.value / 100);
                });
            }
        });
    }

    async togglePlayback(exampleId) {
        const example = this.examples[exampleId];
        
        if (example.isPlaying) {
            this.stopPlayback(exampleId);
        } else {
            // Stop other playing examples
            Object.keys(this.examples).forEach(id => {
                if (id !== exampleId && this.examples[id].isPlaying) {
                    this.stopPlayback(id);
                }
            });
            
            await this.startPlayback(exampleId);
        }
    }

    async startPlayback(exampleId) {
        const example = this.examples[exampleId];
        if (!example.buffer) return;

        // Resume audio context if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }

        // Create source and gain node
        example.source = this.audioContext.createBufferSource();
        example.source.buffer = example.buffer;
        
        example.gainNode = this.audioContext.createGain();
        const volumeSlider = document.getElementById(`${exampleId}-volume`);
        example.gainNode.gain.value = volumeSlider ? volumeSlider.value / 100 : 0.7;
        
        example.source.connect(example.gainNode);
        example.gainNode.connect(this.audioContext.destination);
        
        example.source.onended = () => {
            if (example.isPlaying) {
                this.stopPlayback(exampleId);
            }
        };
        
        example.source.start(0);
        example.isPlaying = true;
        example.startTime = this.audioContext.currentTime;
        
        // Update button to show pause icon
        this.updatePlayButton(exampleId, true);
        
        // Start time update animation
        this.updateTimeDisplay(exampleId);
    }

    stopPlayback(exampleId) {
        const example = this.examples[exampleId];
        
        if (example.source) {
            try {
                example.source.stop();
            } catch (e) {
                // Ignore if already stopped
            }
            example.source = null;
        }
        
        if (example.gainNode) {
            example.gainNode.disconnect();
            example.gainNode = null;
        }
        
        example.isPlaying = false;
        
        // Update button to show play icon
        this.updatePlayButton(exampleId, false);
        
        // Cancel animation frame
        if (this.animationFrames[exampleId]) {
            cancelAnimationFrame(this.animationFrames[exampleId]);
        }
        
        // Reset time display
        document.getElementById(`${exampleId}-current-time`).textContent = '0:00';
        
        // Reset waveform to show no progress
        if (example.buffer) {
            this.drawWaveform(exampleId, example.buffer);
        }
    }

    updatePlayButton(exampleId, isPlaying) {
        const button = document.getElementById(`${exampleId}-play-btn`);
        if (!button) return;
        
        const svg = button.querySelector('svg');
        if (isPlaying) {
            // Pause icon
            svg.innerHTML = '<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>';
        } else {
            // Play icon
            svg.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
        }
    }

    updateTimeDisplay(exampleId) {
        const example = this.examples[exampleId];
        if (!example.isPlaying || !example.buffer) return;
        
        const elapsed = this.audioContext.currentTime - example.startTime;
        const currentTimeEl = document.getElementById(`${exampleId}-current-time`);
        
        if (currentTimeEl) {
            currentTimeEl.textContent = this.formatTime(Math.min(elapsed, example.buffer.duration));
        }
        
        // Update waveform progress
        const progress = Math.min(elapsed / example.buffer.duration, 1);
        this.updateWaveformProgress(exampleId, progress);
        
        if (elapsed < example.buffer.duration) {
            this.animationFrames[exampleId] = requestAnimationFrame(() => this.updateTimeDisplay(exampleId));
        }
    }

    setVolume(exampleId, value) {
        const example = this.examples[exampleId];
        if (example.gainNode) {
            example.gainNode.gain.value = value;
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    stopAllPlayback() {
        Object.keys(this.examples).forEach(id => {
            if (this.examples[id].isPlaying) {
                this.stopPlayback(id);
            }
        });
    }
}

// Initialize Examples Modal
function initializeExamplesModal() {
    const modal = document.getElementById('examples-modal');
    const btn = document.getElementById('examples-btn');
    const closeBtn = document.querySelector('.modal-close-examples');
    const closeFooterBtn = document.querySelector('#examples-modal .modal-btn');
    let examplesPlayer = null;

    if (!modal || !btn) return;

    // Open modal
    btn.addEventListener('click', async function(e) {
        e.preventDefault();
        modal.style.display = 'flex';
        
        // Initialize player if not already done
        if (!examplesPlayer) {
            examplesPlayer = new ExamplesPlayer();
            await examplesPlayer.initialize();
        }
    });

    // Close modal - X button
    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            modal.style.display = 'none';
            if (examplesPlayer) {
                examplesPlayer.stopAllPlayback();
            }
        });
    }

    // Close modal - footer button
    if (closeFooterBtn) {
        closeFooterBtn.addEventListener('click', function() {
            modal.style.display = 'none';
            if (examplesPlayer) {
                examplesPlayer.stopAllPlayback();
            }
        });
    }

    // Close when clicking outside modal content
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
            if (examplesPlayer) {
                examplesPlayer.stopAllPlayback();
            }
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && modal.style.display === 'flex') {
            modal.style.display = 'none';
            if (examplesPlayer) {
                examplesPlayer.stopAllPlayback();
            }
        }
    });
}

// Initialize Contact button
function initializeContactButton() {
    const btn = document.getElementById('contact-btn');
    if (!btn) return;
    
    btn.addEventListener('click', function(e) {
        e.preventDefault();
        const email = 'andrey.vlasenko2006@gmail.com';
        
        // Get button position
        const rect = btn.getBoundingClientRect();
        
        // Create dropdown element
        const dropdown = document.createElement('div');
        dropdown.style.cssText = `
            position: fixed;
            top: ${rect.bottom + 10}px;
            left: ${rect.left}px;
            background: rgba(30, 30, 46, 0.98);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            z-index: 10000;
            animation: slideDown 0.3s ease-out;
            backdrop-filter: blur(10px);
            min-width: 250px;
        `;
        
        dropdown.innerHTML = `
            <div style="color: #e0e0e0; font-size: 14px; margin-bottom: 8px;">Contact Email:</div>
            <div style="color: #6366f1; font-size: 16px; font-weight: 500; user-select: all;">${email}</div>
        `;
        
        // Add animation keyframes if not already present
        if (!document.getElementById('contact-animation-style')) {
            const style = document.createElement('style');
            style.id = 'contact-animation-style';
            style.textContent = `
                @keyframes slideDown {
                    from {
                        opacity: 0;
                        transform: translateY(-10px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
            `;
            document.head.appendChild(style);
        }
        
        document.body.appendChild(dropdown);
        
        // Remove after 3 seconds
        setTimeout(() => {
            dropdown.style.animation = 'slideDown 0.3s ease-out reverse';
            setTimeout(() => dropdown.remove(), 300);
        }, 3000);
        
        // Remove on click anywhere
        setTimeout(() => {
            const removeDropdown = () => {
                dropdown.remove();
                document.removeEventListener('click', removeDropdown);
            };
            document.addEventListener('click', removeDropdown);
        }, 100);
    });
}

// Initialize language selector
function initializeLanguageSelector() {
    const langBtn = document.getElementById('lang-btn');
    const langDropdown = document.getElementById('lang-dropdown');
    const langOptions = document.querySelectorAll('.lang-option');
    
    if (!langBtn || !langDropdown) return;
    
    // Toggle dropdown
    langBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        langBtn.classList.toggle('active');
        langDropdown.classList.toggle('show');
    });
    
    // Language selection
    langOptions.forEach(option => {
        option.addEventListener('click', function(e) {
            e.stopPropagation();
            const lang = this.getAttribute('data-lang');
            
            // Update active state
            langOptions.forEach(opt => opt.classList.remove('active'));
            this.classList.add('active');
            
            // Set language
            setLanguage(lang);
            
            // Close dropdown
            langBtn.classList.remove('active');
            langDropdown.classList.remove('show');
        });
        
        // Set initial active state
        if (option.getAttribute('data-lang') === currentLanguage) {
            option.classList.add('active');
        }
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', function(e) {
        if (!langBtn.contains(e.target) && !langDropdown.contains(e.target)) {
            langBtn.classList.remove('active');
            langDropdown.classList.remove('show');
        }
    });
    
    // Set initial language (async)
    setLanguage(currentLanguage);
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    // Load initial language first
    await setLanguage(currentLanguage);
    
    window.musicLab = new MusicLab();
    window.musicChatbot = new MusicChatbot();
    initializeAboutModal();
    initializeExamplesModal();
    initializeContactButton();
    initializeLanguageSelector();
});
