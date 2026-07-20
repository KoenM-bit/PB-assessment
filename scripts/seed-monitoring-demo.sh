#!/usr/bin/env bash
# Post synthetic /api/predict payloads for monitoring dashboard demos.
# Usage: BASE_URL=https://staging--pb-assessment.netlify.app ./scripts/seed-monitoring-demo.sh [count]
set -euo pipefail

BASE_URL="${BASE_URL:-https://staging--pb-assessment.netlify.app}"
COUNT="${1:-20}"
API="${BASE_URL%/}/api/predict"

regions=("Amsterdam" "Rotterdam" "Utrecht" "The Hague" "Eindhoven" "Groningen" "Maastricht" "Nijmegen")
types=("apartment" "terraced_house" "semi_detached" "detached" "bungalow")
labels=("A" "B" "C" "D" "E")

lat_long() {
  case "$1" in
    Amsterdam) echo "52.3676 4.9041" ;;
    Rotterdam) echo "51.9244 4.4777" ;;
    Utrecht) echo "52.0907 5.1214" ;;
    "The Hague") echo "52.0705 4.3007" ;;
    Eindhoven) echo "51.4416 5.4697" ;;
    Groningen) echo "53.2194 6.5665" ;;
    Maastricht) echo "50.8514 5.6910" ;;
    Nijmegen) echo "51.8426 5.8528" ;;
    *) echo "52.09 5.12" ;;
  esac
}

echo "Seeding ${COUNT} predictions → ${API}"

ok=0
fail=0
for i in $(seq 1 "$COUNT"); do
  region="${regions[$((i % ${#regions[@]}))]}"
  ptype="${types[$((i % ${#types[@]}))]}"
  label="${labels[$((i % ${#labels[@]}))]}"
  read -r lat lon <<<"$(lat_long "$region")"
  # Spread surfaces across training band + a few outliers for drift viz
  if (( i % 7 == 0 )); then
    surface=230
  elif (( i % 11 == 0 )); then
    surface=42
  else
    surface=$((55 + (i * 7) % 145))
  fi
  rooms=$((2 + i % 7))
  bedrooms=$((1 + i % 4))
  year=$((1960 + (i * 3) % 55))

  if (( i % 2 == 0 )); then garden=true; else garden=false; fi

  payload=$(cat <<EOF
{
  "address": "Demo Monitoringstraat ${i}",
  "postcode": "3512 JC",
  "surface_area": ${surface},
  "number_of_rooms": ${rooms},
  "number_of_bedrooms": ${bedrooms},
  "build_year": ${year},
  "energy_label": "${label}",
  "property_type": "${ptype}",
  "garden": ${garden},
  "region": "${region}",
  "latitude": ${lat},
  "longitude": ${lon}
}
EOF
)

  if response=$(curl -fsS -m 120 -X POST "$API" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>&1); then
    price=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['predicted_price'])" 2>/dev/null || echo "?")
    echo "[$i/$COUNT] ok region=${region} surface=${surface}m² → €${price}"
    ok=$((ok + 1))
  else
    echo "[$i/$COUNT] FAILED: $response" >&2
    fail=$((fail + 1))
  fi
  sleep 1
done

echo "Done: ${ok} ok, ${fail} failed"
