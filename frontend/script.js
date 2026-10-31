// frontend/script.js

// 1. Initialize Map centered on Mumbai
const map = L.map('map').setView([19.0760, 72.8777], 11);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap & CartoDB'
}).addTo(map);

// Update slider text in real time
const slider = document.getElementById('budgetSlider');
const budgetDisplay = document.getElementById('budgetValue');
slider.oninput = function() {
    budgetDisplay.innerHTML = "₹" + parseInt(this.value).toLocaleString('en-IN');
}

// Map layer group for wards
let wardLayerGroup = L.layerGroup().addTo(map);

// 2. Fetch Optimization Data
async function runSimulation() {
    const budget = parseInt(slider.value);
    
    try {
        const response = await fetch('http://127.0.0.1:8000/optimize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ budget: budget })
        });
        
        const data = await response.json();
        updateDashboard(data);
        drawMap(data.wards);
    } catch (error) {
        console.error("API Error - Make sure backend is running:", error);
    }
}

// 3. Update Dashboard Stats & Table
function updateDashboard(data) {
    document.getElementById('stat-before').innerText = data.summary.avg_hmri_before;
    document.getElementById('stat-after').innerText = data.summary.avg_hmri_after;
    document.getElementById('stat-reduction').innerText = `-${data.summary.reduction_pct}%`;
    document.getElementById('stat-unspent').innerText = "₹" + data.summary.unspent.toLocaleString('en-IN');

    const tbody = document.getElementById('actions-table');
    tbody.innerHTML = '';
    
    data.wards.slice(0, 8).forEach(ward => {
        if(ward.actions !== "None") {
            const drop = (ward.hmri_before_actions - ward.hmri_after_actions).toFixed(1);
            tbody.innerHTML += `
                <tr class="bg-white">
                    <td class="px-4 py-3 font-bold">${ward.ward_id}</td>
                    <td class="px-4 py-3 text-green-600 font-semibold">-${drop}</td>
                    <td class="px-4 py-3">${ward.actions}</td>
                </tr>
            `;
        }
    });
}

// 4. Draw Heatmap Wards
function drawMap(wardsData) {
    wardLayerGroup.clearLayers();
    
    // Mock mapping logic to spread 24 wards across Mumbai coordinates
    const baseLat = 18.92;
    const baseLon = 72.82;
    
    wardsData.forEach((ward, index) => {
        // Generate pseudo-coordinates for demo map visualization
        const latOffset = (index % 6) * 0.04;
        const lonOffset = Math.floor(index / 6) * 0.04;
        
        // Define a rough square polygon for the ward
        const lat = baseLat + latOffset;
        const lon = baseLon + lonOffset;
        const bounds = [[lat, lon], [lat+0.035, lon], [lat+0.035, lon+0.035], [lat, lon+0.035]];

        // Color intensity based on HMRI *after* actions
        const hmri = ward.hmri_after_actions;
        let color = hmri > 75 ? '#ef4444' : (hmri > 50 ? '#f59e0b' : '#10b981'); // Red -> Orange -> Green

        const polygon = L.polygon(bounds, {
            color: 'white',
            weight: 1,
            fillColor: color,
            fillOpacity: 0.6
        });

        // Clickable popup showing vulnerability breakdown
        const popupContent = `
            <div class="p-2">
                <h3 class="font-bold text-lg mb-1">Ward ${ward.ward_id}</h3>
                <p>Tin Roof Density: <b>${ward.tin_roof_pct.toFixed(0)}%</b></p>
                <p>HMRI Before: <b>${ward.hmri_before_actions.toFixed(1)}</b></p>
                <p>HMRI After: <b>${ward.hmri_after_actions.toFixed(1)}</b></p>
                <hr class="my-2 border-gray-600">
                <p class="text-sm text-yellow-300">Actions: ${ward.actions}</p>
            </div>
        `;
        
        polygon.bindPopup(popupContent, {className: 'custom-popup'});
        wardLayerGroup.addLayer(polygon);
    });
}

// Run initial simulation on page load
window.onload = runSimulation;
