// map object
const map = L.map('bogor_map').setView([-6.6, 106.8], 10);

// add basemap
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 20,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

// choropleth layer
// adapted from: https://leafletjs.com/examples/choropleth/
var dataset;
var geojson;
var colormap;

function getColor(d) {
    return d == 0 ? "0000" : colormap(d);
}

function style(feature) {
    const district = window.dataset.find(x => x.district == feature.properties.NAMOBJ);
    const listingCount = district ? district.count : 0;

    return {
        fillColor: getColor(listingCount),
        weight: 2,
        opacity: 1,
        color: 'white',
        dashArray: '3',
        fillOpacity: 0.7
    };
}

function onEachFeature(feature, layer) {
    layer.on({
        mouseover: highlightFeature,
        mouseout: resetHighlight,
        click: zoomToFeature
    });
}

function highlightFeature(e) {
    var layer = e.target;

    layer.setStyle({
        weight: 5,
        color: '#666',
        dashArray: '',
        fillOpacity: 0.7
    });

    layer.bringToFront();
    info.update(layer.feature.properties);
}

function resetHighlight(e) {
    geojson.resetStyle(e.target);
    info.update();
}

function zoomToFeature(e) {
    map.fitBounds(e.target.getBounds());
}

// popup control
var info = L.control();

info.onAdd = function (map) {
    this._div = L.DomUtil.create('div', 'info'); // create a div with a class "info"
    this.update();
    return this._div;
};

// method that we will use to update the control based on feature properties passed
info.update = function (props) {
    let html = '<h4>Statistik Rumah</h4>';
    if (props) {
        const district = window.dataset.find(x => x.district == props.NAMOBJ);
        if (!district) {
            html += 'Tidak ada data';
            return
        }
        html += '<b>' + district.district + '</b><br />';
        html += 'Jumlah listing: ' + district.count + ' rumah<br />';
        html += 'Median harga: ' + district.price_median.toLocaleString('id-ID', { style: 'currency', currency: 'IDR' });
    } else {
        html += 'Tidak ada data';
    }

    this._div.innerHTML = html;
};

info.addTo(map);

// add legend based on median price
fetch("/api/dashboard/charts/median_price").then(res => res.json()).then(function (data) {
    this.dataset = data;
    this.colormap = chroma.scale(['yellow', '008ae5']).domain([
        Math.min(...data.map(x => x.count)),
        Math.max(...data.map(x => x.count))
    ])

    // create legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function (map) {
        const div = L.DomUtil.create('div', 'info legend');
        const maxCount = Math.max(...data.map(x => x.count));
        const division = 8;
        const intervals = Array.from({ length: division }, (_, i) => Math.ceil((maxCount / division) * i));

        // loop through our density intervals and generate a label with a colored square for each interval
        for (var i = 0; i < intervals.length; i++) {
            div.innerHTML +=
                '<i style="background:' + getColor(intervals[i] + 1) + '"></i> ' +
                intervals[i] + (intervals[i + 1] ? '&ndash;' + intervals[i + 1] + 'jt<br>' : 'jt+');
        }

        return div;
    };

    legend.addTo(map);
}).then(() =>
    // add geojson layer for districts
    fetch("/assets/geojson/bogor.json").then(res => res.json()).then(function (geojson) {
        this.geojson = L.geoJSON(geojson, { style: style, onEachFeature: onEachFeature }).addTo(map);
    })
)

// ------- AMENITIES MAP

// map object
const mapAmenities = L.map('bogor_map_amenities').setView([-6.6, 106.8], 10);

// add basemap
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 20,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(mapAmenities);

// add amenities layer
fetch("/api/dashboard/map/amenities").then(res => res.json()).then(function (points) {
    const amenityColors = {
        "education": "#ffd700",
        "entertainment": "#ffb14e",
        "facilities": "#fa8775",
        "financial": "#ea5f94",
        "healthcare": "#cd34b5",
        "other": "#9d02d7",
        "public_service": "#0000ff",
        "sustenance": "#aee39a",
        "transportation": "#288753"
    }

    function normText(s) {
        const lower = s.toLowerCase().replace('_', ' ')
        return lower.charAt(0).toUpperCase() + lower.slice(1);
    }

    // add markers
    const canvasRenderer = L.canvas({ padding: 0.5, pane: "markerPane" });
    points.forEach(point => L.circleMarker({ lat: point.lat, lng: point.lon }, {
        radius: 5,
        fillOpacity: 0.8,
        renderer: canvasRenderer,
        color: amenityColors[point.category]
    }).bindPopup(`${normText(point.category)} - ${normText(point.amenity)}`).addTo(mapAmenities));

    // create legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function (map) {
        const div = L.DomUtil.create('div', 'info legend');
        Object.keys(amenityColors).sort().forEach(x=>{
            div.innerHTML += `<i style="background:${amenityColors[x]}"></i>${normText(x)}<br>`;
        });

        return div;
    };

    legend.addTo(mapAmenities);
})

