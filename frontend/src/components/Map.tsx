import { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { FeatureCollection } from '../types';
import L from 'leaflet';

// Fix for default marker icon in React Leaflet
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

interface MapProps {
    properties: FeatureCollection | null;
    onBoundsChange?: (bbox: string) => void;
    onSelectProperty?: (id: number) => void;
    city?: string;
}

function MapEvents({ onBoundsChange, city }: { onBoundsChange?: (bbox: string) => void, city?: string }) {
    const map = useMap();

    // Handle City Changes
    useEffect(() => {
        if (!city) return;

        // Coordinates for centers
        const centers: Record<string, [number, number]> = {
            'Barcelona': [41.3851, 2.1734],
            'Valencia': [39.4699, -0.3763],
            'Madrid': [40.4168, -3.7038]
        };

        if (centers[city]) {
            map.setView(centers[city], 13);
        }
    }, [city, map]);

    useEffect(() => {
        if (!onBoundsChange) return;

        const updateBounds = () => {
            const bounds = map.getBounds();
            const bbox = `${bounds.getWest()},${bounds.getSouth()},${bounds.getEast()},${bounds.getNorth()}`;
            onBoundsChange(bbox);
        };

        map.on('moveend', updateBounds);
        map.on('zoomend', updateBounds);

        // Initial load
        updateBounds();

        return () => {
            map.off('moveend', updateBounds);
            map.off('zoomend', updateBounds);
        };
    }, [map, onBoundsChange]);

    return null;
}

export default function Map({ properties, onBoundsChange, onSelectProperty, city }: MapProps) {
    // Barcelona center
    const center: [number, number] = [41.3851, 2.1734];
    const zoom = 13;

    return (
        <div className="h-full w-full">
            <MapContainer center={center} zoom={zoom} scrollWheelZoom={true} className="h-full w-full">
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <MapEvents onBoundsChange={onBoundsChange} city={city} />

                {properties?.features.map((feature) => {
                    const [lon, lat] = feature.geometry.coordinates;
                    return (
                        <Marker
                            key={feature.properties.id}
                            position={[lat, lon]}
                            eventHandlers={{
                                click: () => onSelectProperty && feature.properties.id && onSelectProperty(feature.properties.id)
                            }}
                        >
                        </Marker>
                    )
                })}
            </MapContainer>
        </div>
    );
}
