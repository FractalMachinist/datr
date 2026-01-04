const essayPrompts = [
    "My self summary",
    "What I’m doing with my life",
    "I’m really good at",
    "The first thing people usually notice about me",
    "Favorite books, movies, show, music, and food",
    "The six things I could never do without",
    "I spend a lot of time thinking about",
    "On a typical Friday night I am",
    "The most private thing I am willing to admit",
    "You should message me if..."
];

function loadPreset(type) {
    const presets = {
        engineer: {
            age: 26, sex: 'm', height: 72, orientation: 'straight',
            body_type: 'fit', education: 'graduated from college/university',
            job: 'computer / hardware / software', drinks: 'socially'
        },
        artist: {
            age: 28, sex: 'f', height: 65, orientation: 'bisexual',
            body_type: 'average', education: 'college/university',
            job: 'artistic / musical / writer', drinks: 'rarely'
        },
        student: {
            age: 22, sex: 'f', height: 67, orientation: 'straight',
            body_type: 'thin', education: 'college/university',
            job: 'student', drinks: 'socially'
        }
    };
    const preset = presets[type];
    if (preset) {
        Object.keys(preset).forEach(key => {
            const element = document.getElementById(key);
            if (element) element.value = preset[key];
        });
    }
}

function addEssayToDOM(essay) {
    const essaysDiv = document.getElementById('essays');
    const html = `
        <div class="essay">
            <div class="essay-title">${essay.prompt}</div>
            <div class="essay-content">${essay.text}</div>
        </div>
    `;
    essaysDiv.innerHTML += html;
}

async function generateEssays() {
    const btn = document.getElementById('generateBtn');
    const essaysDiv = document.getElementById('essays');
    btn.disabled = true;
    btn.textContent = '🔄 Generating...';
    essaysDiv.innerHTML = '';
    const profile = {
        age: parseInt(document.getElementById('age').value),
        sex: document.getElementById('sex').value,
        height: parseFloat(document.getElementById('height').value),
        orientation: document.getElementById('orientation').value,
        body_type: document.getElementById('body_type').value,
        education: document.getElementById('education').value,
        job: document.getElementById('job').value,
        drinks: document.getElementById('drinks').value
    };
    try {
        const response = await fetch('/generate_stream', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(profile)
        });
        if (!response.body) throw new Error('No response body');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});
            let boundary;
            while ((boundary = buffer.indexOf('\n')) !== -1) {
                const chunk = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 1);
                if (chunk.trim()) {
                    try {
                        const essay = JSON.parse(chunk);
                        addEssayToDOM(essay);
                    } catch (e) {
                        // Ignore parse errors
                    }
                }
            }
        }
    } catch (error) {
        essaysDiv.innerHTML = `<div class="essay" style="border-left-color: red;">
            <div class="essay-title">Error</div>
            <div class="essay-content">Failed to generate essays: ${error.message}</div>
        </div>`;
    }
    btn.disabled = false;
    btn.textContent = '✨ Generate Profile Essays';
}
