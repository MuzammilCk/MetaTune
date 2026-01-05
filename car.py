import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

print("🚗 Generating GLOBAL Car Dataset for MetaTune...")

# 1. Comprehensive Regional & Global Models
MODELS_BY_BRAND = {
    # JAPAN
    'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Tacoma', 'Tundra', 'Prius', 'Supra', '4Runner', 'Sienna', 'Land Cruiser', 'Sequoia', 'Venza', 'Crown', 'Mirai', 'GR86', 'Avalon', 'C-HR', 'bZ4X', 'Yaris', 'Aygo', 'Alphard', 'Hilux', 'Fortuner'],
    'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot', 'Odyssey', 'Ridgeline', 'HR-V', 'Passport', 'Fit', 'Insight', 'Prologue', 'Element', 'Civic Type R', 'Jazz', 'City', 'Amaze', 'ZR-V'],
    'Nissan': ['Altima', 'Sentra', 'Rogue', 'Pathfinder', 'Murano', 'Armada', 'Frontier', 'Titan', 'Leaf', 'Versa', 'Maxima', 'Kicks', 'GT-R', 'Z', 'Ariya', 'Qashqai', 'X-Trail', 'Navara', 'Patrol', 'Juke'],
    'Mazda': ['Mazda3', 'Mazda6', 'CX-5', 'CX-30', 'CX-50', 'CX-90', 'MX-5 Miata', 'CX-9', 'MX-30', 'CX-70', 'CX-60', 'BT-50', 'Mazda2'],
    'Subaru': ['Outback', 'Forester', 'Crosstrek', 'Impreza', 'Legacy', 'Ascent', 'WRX', 'BRZ', 'Solterra', 'Levorg'],
    'Mitsubishi': ['Outlander', 'Outlander Sport', 'Eclipse Cross', 'Mirage', 'Mirage G4', 'Outlander PHEV', 'Pajero', 'Triton', 'Xpander', 'Delica'],
    'Suzuki': ['Swift', 'Vitara', 'Jimny', 'S-Cross', 'Ignis', 'Across', 'Swace', 'Baleno', 'Ertiga', 'Celerio', 'Dzire', 'Alto'],
    'Lexus': ['ES', 'IS', 'LS', 'NX', 'RX', 'GX', 'LX', 'UX', 'RC', 'LC', 'RZ', 'IS 500', 'RC F', 'LBX', 'LM'],
    'Infiniti': ['Q50', 'Q60', 'QX50', 'QX55', 'QX60', 'QX80'],
    'Acura': ['TLX', 'Integra', 'RDX', 'MDX', 'ZDX', 'NSX'],
    'Isuzu': ['D-Max', 'MU-X'],

    # USA
    'Ford': ['F-150', 'Mustang', 'Explorer', 'Escape', 'Edge', 'Expedition', 'Ranger', 'Bronco', 'Maverick', 'Transit', 'F-250 Super Duty', 'Focus', 'Fiesta', 'Taurus', 'Bronco Sport', 'Mustang Mach-E', 'Puma', 'Kuga', 'Everest', 'Mondeo'],
    'Chevrolet': ['Silverado', 'Equinox', 'Malibu', 'Traverse', 'Tahoe', 'Suburban', 'Colorado', 'Blazer', 'Camaro', 'Corvette', 'Trax', 'Bolt EV', 'Impala', 'Bolt EUV', 'Silverado EV', 'Trailblazer', 'Montana', 'S10'],
    'Jeep': ['Wrangler', 'Grand Cherokee', 'Cherokee', 'Compass', 'Renegade', 'Gladiator', 'Wagoneer', 'Grand Wagoneer', 'Avenger', 'Commander'],
    'Tesla': ['Model S', 'Model 3', 'Model X', 'Model Y', 'Cybertruck', 'Roadster'],
    'GMC': ['Sierra 1500', 'Sierra 2500HD', 'Canyon', 'Terrain', 'Acadia', 'Yukon', 'Yukon XL', 'Hummer EV'],
    'Dodge': ['Charger', 'Challenger', 'Durango', 'Hornet', 'Journey', 'Grand Caravan'],
    'Ram': ['1500', '2500', '3500', 'ProMaster', '1500 TRX', 'Rampage'],
    'Cadillac': ['CT4', 'CT5', 'XT4', 'XT5', 'XT6', 'Escalade', 'Lyriq', 'Celestiq', 'Optiq'],
    'Buick': ['Encore', 'Encore GX', 'Envision', 'Enclave', 'Envista'],
    'Lincoln': ['Navigator', 'Aviator', 'Nautilus', 'Corsair', 'Z'],
    'Chrysler': ['300', 'Pacifica', 'Voyager'],
    'Rivian': ['R1T', 'R1S', 'R2'],
    'Lucid': ['Air', 'Gravity'],
    'Fisker': ['Ocean', 'Pear', 'Alaska'],

    # GERMANY
    'BMW': ['3 Series', '5 Series', '7 Series', 'X1', 'X3', 'X5', 'X7', 'Z4', 'M3', 'M4', 'M5', 'i4', 'iX', 'i7', '8 Series', 'X2', 'X4', 'X6', 'M2', 'iX1', 'iX3', 'XM'],
    'Mercedes-Benz': ['C-Class', 'E-Class', 'S-Class', 'A-Class', 'CLA', 'GLA', 'GLB', 'GLC', 'GLE', 'GLS', 'G-Class', 'AMG GT', 'EQS', 'EQE', 'SL-Class', 'B-Class', 'V-Class', 'EQA', 'EQB', 'EQC'],
    'Audi': ['A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Q3', 'Q5', 'Q7', 'Q8', 'e-tron', 'e-tron GT', 'TT', 'R8', 'RS6', 'Q2', 'Q4 e-tron', 'Q6 e-tron'],
    'Volkswagen': ['Jetta', 'Passat', 'Golf', 'Tiguan', 'Atlas', 'ID.4', 'Taos', 'Arteon', 'Beetle', 'Touareg', 'Golf GTI', 'Golf R', 'Polo', 'T-Roc', 'T-Cross', 'ID.3', 'ID.7', 'Amarok'],
    'Porsche': ['911', 'Cayenne', 'Macan', 'Panamera', 'Taycan', '718 Boxster', '718 Cayman'],

    # SOUTH KOREA
    'Hyundai': ['Elantra', 'Sonata', 'Tucson', 'Santa Fe', 'Palisade', 'Kona', 'Venue', 'Ioniq 5', 'Ioniq 6', 'Nexo', 'Santa Cruz', 'Creta', 'i10', 'i20', 'i30', 'Staria', 'Casper'],
    'Kia': ['Forte', 'K5', 'Sportage', 'Sorento', 'Telluride', 'Soul', 'Seltos', 'Carnival', 'EV6', 'EV9', 'Niro', 'Rio', 'Stinger', 'Picanto', 'Sonet', 'Carens', 'EV5'],
    'Genesis': ['G70', 'G80', 'G90', 'GV60', 'GV70', 'GV80'],

    # UK
    'Land Rover': ['Range Rover', 'Range Rover Sport', 'Range Rover Velar', 'Range Rover Evoque', 'Defender', 'Discovery', 'Discovery Sport'],
    'Jaguar': ['F-PACE', 'E-PACE', 'I-PACE', 'F-TYPE', 'XE', 'XF'],
    'Mini': ['Cooper', 'Cooper S', 'Countryman', 'Clubman', 'Convertible', 'Aceman'],
    'Bentley': ['Continental GT', 'Flying Spur', 'Bentayga'],
    'Rolls-Royce': ['Phantom', 'Ghost', 'Cullinan', 'Spectre'],
    'Aston Martin': ['DB12', 'Vantage', 'DBX', 'Valhalla'],
    'McLaren': ['750S', 'Artura', 'GTS'],
    'Lotus': ['Emira', 'Eletre', 'Emeya'],

    # ITALY
    'Ferrari': ['Roma', '296 GTB', 'SF90', 'Purosangue', '12Cilindri'],
    'Lamborghini': ['Huracán', 'Revuelto', 'Urus'],
    'Maserati': ['Grecale', 'Levante', 'Ghibli', 'Quattroporte', 'MC20', 'GranTurismo'],
    'Alfa Romeo': ['Giulia', 'Stelvio', 'Tonale', 'Milano'],
    'Fiat': ['500', '500e', 'Panda', 'Tipo', 'Pulse', 'Fastback', 'Strada', 'Titano'],
    'Lancia': ['Ypsilon'],

    # FRANCE
    'Renault': ['Clio', 'Captur', 'Megane', 'Arkana', 'Austral', 'Zoe', 'Twingo', 'Espace', 'Rafale', 'Scenic', 'Kangoo', 'Kwid'],
    'Peugeot': ['208', '2008', '308', '3008', '508', '5008', '408', 'Landtrek', 'Rifter', 'Traveller'],
    'Citroen': ['C3', 'C3 Aircross', 'C4', 'C5 Aircross', 'C5 X', 'Berlingo', 'Ami'],
    'Bugatti': ['Chiron', 'Mistral', 'Bolide'],
    'Alpine': ['A110'],
    'DS': ['DS 3', 'DS 4', 'DS 7', 'DS 9'],

    # SWEDEN
    'Volvo': ['XC40', 'XC60', 'XC90', 'S60', 'S90', 'V60', 'V90', 'C40', 'EX30', 'EX90', 'EM90'],
    'Polestar': ['2', '3', '4'],
    'Koenigsegg': ['Jesko', 'Gemera'],

    # CHINA (Major Expansion)
    'BYD': ['Seal', 'Dolphin', 'Atto 3', 'Han', 'Tang', 'Qin Plus', 'Song Plus', 'Seagull', 'Yuan Plus', 'Chaser 05'],
    'Geely': ['Coolray', 'Monjaro', 'Tugella', 'Emgrand', 'Okavango', 'Azkarra', 'Preface'],
    'NIO': ['ET7', 'ET5', 'ES8', 'ES6', 'EC7', 'EC6', 'EL7'],
    'XPeng': ['P7', 'P5', 'G9', 'G6', 'G3i'],
    'MG': ['MG4', 'ZS', 'HS', 'Hector', 'Gloster', 'Comet', 'Cyberster', 'MG5', 'Marvel R'],
    'Zeekr': ['001', '009', 'X', '007'],
    'Hongqi': ['H9', 'E-HS9', 'H5'],
    'Chery': ['Tiggo 8', 'Tiggo 7', 'Arrizo 8', 'Omoda 5'],
    'GWM': ['Haval H6', 'Tank 300', 'Tank 500', 'Ora Funky Cat', 'Poer Cannon'],

    # INDIA
    'Tata': ['Nexon', 'Harrier', 'Safari', 'Punch', 'Tiago', 'Tigor', 'Altroz', 'Curvv'],
    'Mahindra': ['XUV700', 'Scorpio-N', 'Thar', 'XUV300', 'Bolero', 'Marazzo'],

    # SPAIN/CZECH/OTHER
    'Seat': ['Ibiza', 'Leon', 'Arona', 'Ateca', 'Tarraco'],
    'Cupra': ['Formentor', 'Born', 'Tavascan', 'Leon', 'Ateca'],
    'Skoda': ['Octavia', 'Superb', 'Kodiaq', 'Karoq', 'Kamiq', 'Fabia', 'Scala', 'Enyaq', 'Kushaq', 'Slavia'],
    'Dacia': ['Duster', 'Sandero', 'Jogger', 'Spring'],
    'Holden': ['Commodore', 'Colorado', 'Acadia', 'Equinox']
}

BRAND_ORIGIN = {
    'Toyota': 'Japan', 'Honda': 'Japan', 'Nissan': 'Japan', 'Mazda': 'Japan', 'Subaru': 'Japan', 
    'Mitsubishi': 'Japan', 'Suzuki': 'Japan', 'Lexus': 'Japan', 'Infiniti': 'Japan', 'Acura': 'Japan', 'Isuzu': 'Japan',
    'Ford': 'USA', 'Chevrolet': 'USA', 'Jeep': 'USA', 'Tesla': 'USA', 'GMC': 'USA', 'Dodge': 'USA', 
    'Ram': 'USA', 'Cadillac': 'USA', 'Buick': 'USA', 'Lincoln': 'USA', 'Chrysler': 'USA', 
    'Rivian': 'USA', 'Lucid': 'USA', 'Fisker': 'USA',
    'BMW': 'Germany', 'Mercedes-Benz': 'Germany', 'Audi': 'Germany', 'Volkswagen': 'Germany', 'Porsche': 'Germany',
    'Hyundai': 'South Korea', 'Kia': 'South Korea', 'Genesis': 'South Korea',
    'Land Rover': 'UK', 'Jaguar': 'UK', 'Mini': 'UK', 'Bentley': 'UK', 'Rolls-Royce': 'UK', 
    'Aston Martin': 'UK', 'McLaren': 'UK', 'Lotus': 'UK',
    'Ferrari': 'Italy', 'Lamborghini': 'Italy', 'Maserati': 'Italy', 'Alfa Romeo': 'Italy', 
    'Fiat': 'Italy', 'Lancia': 'Italy',
    'Renault': 'France', 'Peugeot': 'France', 'Citroen': 'France', 'Bugatti': 'France', 'Alpine': 'France', 'DS': 'France',
    'Volvo': 'Sweden', 'Polestar': 'Sweden', 'Koenigsegg': 'Sweden',
    'BYD': 'China', 'Geely': 'China', 'NIO': 'China', 'XPeng': 'China', 'MG': 'China', 
    'Zeekr': 'China', 'Hongqi': 'China', 'Chery': 'China', 'GWM': 'China',
    'Tata': 'India', 'Mahindra': 'India',
    'Seat': 'Spain', 'Cupra': 'Spain',
    'Skoda': 'Czech Republic',
    'Dacia': 'Romania',
    'Holden': 'Australia'
}

BRANDS = list(MODELS_BY_BRAND.keys())
# Define body types and colors
BODY_TYPES = ['Sedan', 'SUV', 'Truck', 'Coupe', 'Hatchback', 'Wagon', 'Van', 'Convertible', 'Crossover', 'Minivan']
COLORS = ['Black', 'White', 'Silver', 'Gray', 'Red', 'Blue', 'Green', 'Yellow', 'Orange', 'Brown', 'Gold', 'Purple', 'Beige']

# Keywords to enforce realistic body types
SUV_KEYWORDS = ['RX', 'GX', 'LX', 'NX', 'X1', 'X3', 'X5', 'X7', 'Q3', 'Q5', 'Q7', 'Q8', 'GLA', 'GLB', 'GLC', 'GLE', 'GLS', 'G-Class', 'Cayenne', 'Macan', 'Urus', 'Purosangue', 'Bentayga', 'Cullinan', 'CR-V', 'RAV4', 'Highlander', 'Pilot', 'Explorer', 'Tahoe', 'Suburban', 'Yukon', 'Escalade', 'Pathfinder', 'Armada', 'Telluride', 'Palisade', 'Santa Fe', 'Tucson', 'Sportage', 'Sorento', 'CX-5', 'CX-9', 'Outback', 'Forester', 'Ascent', 'Wrangler', 'Cherokee', 'Grand Cherokee', 'Durango', 'Enclave', 'XT4', 'XT5', 'XT6', 'Aviator', 'Navigator', 'MDX', 'RDX', 'QX50', 'QX60', 'QX80', 'GV70', 'GV80', 'Model Y', 'Model X', 'I-PACE', 'F-PACE', 'E-PACE', 'Range Rover', 'Defender', 'Discovery', 'XC40', 'XC60', 'XC90', '3008', '5008', 'Captur', 'Koleos', '500X', 'Stelvio', 'Tonale', 'Outlander', 'Eclipse Cross', 'Vitara', 'S-Cross', 'MU-X', 'Pacifica', 'Voyager', 'Carnival', 'Odyssey', 'Sienna', 'Tang', 'Song', 'Atto', 'Nexon', 'Harrier', 'Safari', 'XUV', 'Thar', 'Scorpio', 'Hector', 'Gloster', 'Tiggo', 'Haval', 'Tank', 'Coolray', 'Monjaro', 'Tugella', 'Atlas', 'Tiguan', 'Kodiaq', 'Karoq', 'Kamiq', 'Ateca', 'Tarraco', 'Duster', 'Jogger']
TRUCK_KEYWORDS = ['F-150', 'F-250', 'Silverado', 'Sierra', 'Ram', 'Tundra', 'Tacoma', 'Titan', 'Frontier', 'Colorado', 'Canyon', 'Ranger', 'Maverick', 'Ridgeline', 'Gladiator', 'Santa Cruz', 'Cybertruck', 'Hummer EV', 'Rivian', 'D-Max', 'Hilux', 'Navara', 'Triton', 'Amarok', 'Cannon', 'Landtrek', 'Strada', 'Titano', 'Poer']
COUPE_KEYWORDS = ['Mustang', 'Camaro', 'Corvette', 'Challenger', 'Supra', 'GR86', 'BRZ', 'Miata', '911', '718', 'Taycan', 'R8', 'TT', 'AMG GT', 'SL-Class', 'F-TYPE', 'LC', 'RC', 'Z', 'GT-R', '488', 'F8', 'Roma', 'SF90', 'Huracán', 'Aventador', 'Revuelto', '4C', 'Beetle', 'M4', 'M8', '8 Series', 'Model S', 'Roadster', 'Spectre', 'Chiron', 'Mistral', 'Bolide', 'Jesko', 'Gemera', 'Valhalla', 'Emira', 'Cyberster']
HATCH_KEYWORDS = ['Golf', 'Civic', 'Mazda3', 'Corolla', 'Impreza', 'Fit', 'Yaris', 'Rio', 'Spark', 'Sonic', 'Bolt', 'Leaf', 'Mini', 'Cooper', '208', '308', 'Clio', 'Megane', 'Panda', '500', 'Swift', 'Ignis', 'Yaris', 'Aygo', 'Polo', 'Fabia', 'Ibiza', 'Sandero', 'Spring', 'Dolphin', 'Seagull', 'MG4', 'Ora', 'Tiago', 'Altroz', 'Kwid', 'C3']

n_cars = 15000

data = {
    'car_id': [f'CAR-{str(i).zfill(6)}' for i in range(1, n_cars + 1)],
    'brand': [],
    'model': [],
    'origin_country': [],
    'year': np.random.randint(2013, 2026, n_cars),
    'body_type': [],
    'fuel_type': [],
    'transmission': [],
    'drive_type': [],
    'engine_size_L': [],
    'cylinders': [],
    'horsepower': [],
    'torque_lb_ft': [],
    'fuel_economy_combined_mpg': [],
    'price_usd': [],
    'mileage_km': [],
    'condition': [],
    'color': np.random.choice(COLORS, n_cars),
    'seating_capacity': [],
    'doors': [],
    'market_segment': [],
    'owner_satisfaction': [],
    'reliability_score': []
}

# 2. Real-World Specs Lookup (Hybrid Approach)
# For top sellers, we use HARDCODED real stats instead of random generation.
REAL_WORLD_SPECS = {
    # Toyota
    'Corolla': {'body': 'Sedan', 'eng': 2.0, 'cyl': 4, 'hp': 169, 'tq': 151, 'mpg': 35},
    'Camry': {'body': 'Sedan', 'eng': 2.5, 'cyl': 4, 'hp': 203, 'tq': 184, 'mpg': 32},
    'RAV4': {'body': 'SUV', 'eng': 2.5, 'cyl': 4, 'hp': 203, 'tq': 184, 'mpg': 30},
    'Prius': {'body': 'Hatchback', 'eng': 2.0, 'cyl': 4, 'hp': 194, 'tq': 139, 'mpg': 57},
    'Supra': {'body': 'Coupe', 'eng': 3.0, 'cyl': 6, 'hp': 382, 'tq': 368, 'mpg': 26},
    'Tacoma': {'body': 'Truck', 'eng': 2.4, 'cyl': 4, 'hp': 278, 'tq': 317, 'mpg': 21},
    
    # Honda
    'Civic': {'body': 'Sedan', 'eng': 1.5, 'cyl': 4, 'hp': 180, 'tq': 177, 'mpg': 36},
    'Accord': {'body': 'Sedan', 'eng': 1.5, 'cyl': 4, 'hp': 192, 'tq': 192, 'mpg': 32},
    'CR-V': {'body': 'SUV', 'eng': 1.5, 'cyl': 4, 'hp': 190, 'tq': 179, 'mpg': 30},
    'Civic Type R': {'body': 'Hatchback', 'eng': 2.0, 'cyl': 4, 'hp': 315, 'tq': 310, 'mpg': 24},

    # Ford
    'F-150': {'body': 'Truck', 'eng': 3.5, 'cyl': 6, 'hp': 400, 'tq': 500, 'mpg': 20},
    'Mustang': {'body': 'Coupe', 'eng': 5.0, 'cyl': 8, 'hp': 480, 'tq': 415, 'mpg': 18},
    'Explorer': {'body': 'SUV', 'eng': 2.3, 'cyl': 4, 'hp': 300, 'tq': 310, 'mpg': 24},
    'Bronco': {'body': 'SUV', 'eng': 2.7, 'cyl': 6, 'hp': 330, 'tq': 415, 'mpg': 17},

    # Tesla (Electric)
    'Model 3': {'body': 'Sedan', 'eng': 0.0, 'cyl': 0, 'hp': 283, 'tq': 302, 'mpg': 132}, # RWD base
    'Model Y': {'body': 'SUV', 'eng': 0.0, 'cyl': 0, 'hp': 384, 'tq': 376, 'mpg': 125},   # Long Range
    'Model S': {'body': 'Sedan', 'eng': 0.0, 'cyl': 0, 'hp': 670, 'tq': 755, 'mpg': 120},
    'Cybertruck': {'body': 'Truck', 'eng': 0.0, 'cyl': 0, 'hp': 600, 'tq': 740, 'mpg': 85},

    # Chevrolet
    'Silverado': {'body': 'Truck', 'eng': 5.3, 'cyl': 8, 'hp': 355, 'tq': 383, 'mpg': 18},
    'Corvette': {'body': 'Coupe', 'eng': 6.2, 'cyl': 8, 'hp': 490, 'tq': 465, 'mpg': 19},
    'Tahoe': {'body': 'SUV', 'eng': 5.3, 'cyl': 8, 'hp': 355, 'tq': 383, 'mpg': 17},

    # Ram
    '1500': {'body': 'Truck', 'eng': 5.7, 'cyl': 8, 'hp': 395, 'tq': 410, 'mpg': 19},
    '1500 TRX': {'body': 'Truck', 'eng': 6.2, 'cyl': 8, 'hp': 702, 'tq': 650, 'mpg': 12},

    # BMW
    '3 Series': {'body': 'Sedan', 'eng': 2.0, 'cyl': 4, 'hp': 255, 'tq': 295, 'mpg': 29},
    'M3': {'body': 'Sedan', 'eng': 3.0, 'cyl': 6, 'hp': 473, 'tq': 406, 'mpg': 19},
    'X5': {'body': 'SUV', 'eng': 3.0, 'cyl': 6, 'hp': 375, 'tq': 398, 'mpg': 25},

    # Nissan
    'Altima': {'body': 'Sedan', 'eng': 2.5, 'cyl': 4, 'hp': 188, 'tq': 180, 'mpg': 32},
    'GT-R': {'body': 'Coupe', 'eng': 3.8, 'cyl': 6, 'hp': 565, 'tq': 467, 'mpg': 18},    

    # Porsche
    '911': {'body': 'Coupe', 'eng': 3.0, 'cyl': 6, 'hp': 379, 'tq': 331, 'mpg': 20},
    'Macan': {'body': 'SUV', 'eng': 2.0, 'cyl': 4, 'hp': 261, 'tq': 295, 'mpg': 21},

    # Volkswagen
    'Golf GTI': {'body': 'Hatchback', 'eng': 2.0, 'cyl': 4, 'hp': 241, 'tq': 273, 'mpg': 28},
    'Jetta': {'body': 'Sedan', 'eng': 1.5, 'cyl': 4, 'hp': 158, 'tq': 184, 'mpg': 35},

    # Subaru
    'Outback': {'body': 'SUV', 'eng': 2.5, 'cyl': 4, 'hp': 182, 'tq': 176, 'mpg': 28},
    'WRX': {'body': 'Sedan', 'eng': 2.4, 'cyl': 4, 'hp': 271, 'tq': 258, 'mpg': 22},

    # Jeep
    'Wrangler': {'body': 'SUV', 'eng': 3.6, 'cyl': 6, 'hp': 285, 'tq': 260, 'mpg': 19},
}

for i in range(n_cars):
    # 1. Brand/Model/Origin
    brand = np.random.choice(BRANDS)
    # Validate key existence to assume no 'Generic' fallback needs
    if brand not in MODELS_BY_BRAND:
        continue # Should not happen
        
    model = np.random.choice(MODELS_BY_BRAND[brand])
    origin = BRAND_ORIGIN[brand] # Strict dictionary access
    
    data['brand'].append(brand)
    data['model'].append(model)
    data['origin_country'].append(origin)
    
    # Check if we have REAL WORLD data for this model
    real_spec = REAL_WORLD_SPECS.get(model)
    
    # 2. Body Type
    if real_spec:
        b_type = real_spec['body']
        if b_type == 'Truck': base_price_mod = 1.3
        elif b_type == 'SUV': base_price_mod = 1.2
        elif b_type in ['Coupe', 'Convertible']: base_price_mod = 1.5
        elif b_type == 'Hatchback': base_price_mod = 0.8
        else: base_price_mod = 1.0
        
    else:
        # Fallback to intelligent guessing for models not in Top 50
        if any(k in model for k in TRUCK_KEYWORDS):
            b_type = 'Truck'
            base_price_mod = 1.3
        elif any(k in model for k in SUV_KEYWORDS) or 'SUV' in model:
            b_type = 'SUV'
            base_price_mod = 1.2
        elif any(k in model for k in COUPE_KEYWORDS):
            b_type = np.random.choice(['Coupe', 'Convertible'], p=[0.7, 0.3])
            base_price_mod = 1.5
        elif any(k in model for k in HATCH_KEYWORDS):
            b_type = 'Hatchback'
            base_price_mod = 0.8
        elif 'Van' in model or 'Minivan' in model:
            b_type = 'Minivan'
            base_price_mod = 1.1
        else:
            b_type = 'Sedan'
            base_price_mod = 1.0
            
    data['body_type'].append(b_type)

    # 3. Engine & Performance
    year = data['year'][i]
    is_electric = False
    electric_models = ['Model', 'Bolt', 'Leaf', 'Ioniq', 'EV6', 'Taycan', 'e-tron', 'EQS', 'Hummer EV', 'Lyriq', 'Solterra', 'ID.', 'Zoe', '500e', 'Spring', 'Seagull', 'Dolphin', 'Seal', 'Atto', 'Han', 'Tang', 'NIO', 'XPeng', 'MG4', 'ZS EV', 'Comet', 'Cyberster', 'Zeekr', 'Spectre', 'Rivian', 'Lucid', 'Fisker', 'Mustang Mach-E', 'F-150 Lightning', 'Silverado EV', 'Blazer EV', 'Equinox EV', 'Polestar', 'bZ4X', 'Ariya', 'Solterra', 'GV60', 'Electrified']

    if base_price_mod: pass # dummy usage

    if real_spec:
        # USE REAL DATA
        fuel = 'Electric' if real_spec['eng'] == 0.0 else 'Gasoline' 
        # (Simplified, could add specific Hybrid logic if needed in lookup)
        if model in ['Prius']: fuel = 'Hybrid'
        
        cylinders = real_spec['cyl']
        displacement = real_spec['eng']
        hp = int(real_spec['hp'] * np.random.uniform(0.95, 1.05)) # Tiny variance for uniqueness
        tq = int(real_spec['tq'] * np.random.uniform(0.95, 1.05))
        if displacement == 0.0: is_electric = True
        
    else:
        # PROCEDURAL FALLBACK
        if brand in ['Tesla', 'Rivian', 'Lucid', 'Fisker', 'Polestar'] or any(em in model for em in electric_models):
            fuel = 'Electric'
            is_electric = True
            cylinders = 0
            displacement = 0.0
            hp = np.random.randint(150, 1000)
        elif brand in ['Ferrari', 'Lamborghini', 'McLaren', 'Bugatti', 'Koenigsegg']:
            fuel = 'Gasoline'
            cylinders = np.random.choice([8, 10, 12, 16])
            displacement = np.random.uniform(3.8, 8.0)
            hp = np.random.randint(600, 1500)
        elif b_type == 'Truck':
            fuel = np.random.choice(['Gasoline', 'Diesel'], p=[0.7, 0.3])
            cylinders = np.random.choice([6, 8])
            displacement = np.random.uniform(2.7, 6.7)
            hp = np.random.randint(280, 500)
        elif brand in ['BYD', 'Geely', 'Chery', 'MG'] and not is_electric:
            fuel = 'Gasoline'
            cylinders = 4
            displacement = np.random.uniform(1.5, 2.0)
            hp = np.random.randint(130, 250)
        elif origin == 'India' and not is_electric:
            fuel = np.random.choice(['Gasoline', 'Diesel'], p=[0.6, 0.4])
            cylinders = np.random.choice([3, 4])
            displacement = np.random.uniform(1.0, 2.2)
            hp = np.random.randint(80, 180)
        else:
            fuel = np.random.choice(['Gasoline', 'Hybrid', 'Diesel'], p=[0.75, 0.15, 0.1])
            if fuel == 'Hybrid':
                cylinders = 4
                displacement = np.random.uniform(1.5, 2.5)
                hp = np.random.randint(120, 300)
            else:
                cylinders = np.random.choice([3, 4, 6, 8], p=[0.15, 0.55, 0.25, 0.05])
                displacement = cylinders * 0.5 + np.random.uniform(-0.2, 0.2)
                hp = int(displacement * 65 + np.random.uniform(20, 80))
        
        tq = int(hp * np.random.uniform(0.9, 1.5))

    data['fuel_type'].append(fuel)
    data['cylinders'].append(cylinders)
    data['engine_size_L'].append(round(displacement, 1))
    data['horsepower'].append(hp)
    data['torque_lb_ft'].append(tq)

    # 4. Transmission
    if is_electric:
        trans = 'Automatic'
    elif brand in ['Ferrari', 'Lamborghini', 'Porsche', 'McLaren']:
        trans = 'Dual-Clutch'
    elif brand in ['Toyota', 'Honda', 'Nissan', 'Subaru'] and fuel == 'Gasoline':
        trans = np.random.choice(['CVT', 'Automatic', 'Manual'], p=[0.5, 0.4, 0.1])
    elif origin == 'Europe':
        trans = np.random.choice(['Automatic', 'Manual', 'Dual-Clutch'], p=[0.5, 0.3, 0.2])
    else:
        trans = 'Automatic'
    data['transmission'].append(trans)

    # 5. Drive Type
    if b_type in ['Truck', 'SUV']:
        drive = np.random.choice(['AWD', '4WD', 'RWD'], p=[0.4, 0.4, 0.2])
    elif brand in ['Audi', 'Subaru']:
        drive = 'AWD'
    elif brand in ['BMW', 'Mercedes-Benz']:
        drive = np.random.choice(['RWD', 'AWD'], p=[0.5, 0.5])
    else:
        drive = np.random.choice(['FWD', 'AWD'], p=[0.8, 0.2])
    data['drive_type'].append(drive)

    # 6. Mileage & Condition
    age = 2026 - year
    mileage = int(age * np.random.uniform(5000, 25000))
    data['mileage_km'].append(mileage)
    if mileage < 1000: condition = 'New'
    elif mileage < 40000: condition = 'Used - Excellent'
    elif mileage < 100000: condition = 'Used - Good'
    else: condition = 'Used - Fair'
    data['condition'].append(condition)

    # 7. Price Calculation
    # Baseline by Brand Tier
    tier_base = {
        'Ultra-Luxury': 250000, 'Luxury': 70000, 'Premium': 45000, 
        'Mid-Range': 28000, 'Economy': 18000, 'Budget': 12000
    }
    
    if brand in ['Ferrari', 'Lamborghini', 'Bugatti', 'Koenigsegg', 'McLaren', 'Rolls-Royce', 'Bentley', 'Aston Martin']:
        tier = 'Ultra-Luxury'
    elif brand in ['Porsche', 'Mercedes-Benz', 'BMW', 'Audi', 'Land Rover', 'Lexus', 'Lucid', 'Lotus']:
        tier = 'Luxury'
    elif brand in ['Volvo', 'Tesla', 'Genesis', 'Cadillac', 'Lincoln', 'Acura', 'Infiniti', 'Alfa Romeo', 'Jaguar', 'Rivian', 'NIO', 'Zeekr', 'Hongqi', 'Polestar']:
        tier = 'Premium'
    elif brand in ['Toyota', 'Honda', 'Ford', 'Volkswagen', 'Mazda', 'Subaru', 'Jeep', 'GMC', 'Kia', 'Hyundai', 'BYD', 'XPeng', 'Peugeot', 'Skoda', 'Cupra', 'Mini']:
        tier = 'Mid-Range'
    elif brand in ['Dacia', 'Suzuki', 'Fiat', 'Renault', 'Citroen', 'Seat', 'MG', 'Chery', 'GWM', 'Geely', 'Mahindra', 'Tata']:
        tier = 'Economy'
    else:
        tier = 'Economy'
        
    # Override for Budget models
    if b_type == 'Hatchback' and tier == 'Economy':
        tier = 'Budget'

    base_p = tier_base[tier] * base_price_mod
    
    # HP Premium
    price = base_p + (hp * 50)
    
    # Depreciation
    dep_rate = 0.88 if brand in ['Toyota', 'Lexus', 'Porsche'] else 0.82
    final_price = int(price * (dep_rate ** age))
    
    data['price_usd'].append(max(final_price, 1500))
    data['market_segment'].append(tier)

    # 8. Reliability & Satisfaction
    rel = 75
    if origin == 'Japan': rel += 10
    if origin == 'South Korea': rel += 5
    if brand in ['Land Rover', 'Jaguar', 'Alfa Romeo', 'Fiat']: rel -= 15
    data['reliability_score'].append(int(np.clip(rel + np.random.normal(0, 5), 0, 100)))
    data['owner_satisfaction'].append(round(np.random.uniform(3.0, 5.0), 1))

    # 9. Practicality
    seats = 5
    doors = 4
    if b_type in ['Coupe', 'Convertible']: seats=2; doors=2
    elif b_type in ['SUV', 'Minivan'] and np.random.rand() > 0.6: seats=7
    data['seating_capacity'].append(seats)
    data['doors'].append(doors)

    # 10. Fuel Economy
    if real_spec:
        mpg = real_spec['mpg']
        # Adust for age slightly
        if age > 5: mpg -= 2
    elif is_electric:
        mpg = np.random.randint(90, 140)
    else:
        mpg = int(50 - (hp/25))
        if fuel == 'Hybrid': mpg += 15
        if fuel == 'Diesel': mpg += 8
    data['fuel_economy_combined_mpg'].append(max(8, mpg))

# To DataFrame
df = pd.DataFrame(data)
df.to_csv('global_car_dataset.csv', index=False)

print(f"\n✅ GLOBAL Dataset Generated Successfully!")
print(f"   Total Cars: {len(df):,}")
print(f"   Unique Brands: {df['brand'].nunique()}")
print(f"   Unique Models: {df['model'].nunique()}")
print(f"   Countries of Origin: {df['origin_country'].unique()}")
print(f"   Market Segments: {df['market_segment'].value_counts().to_dict()}")
print(f"\n💾 Saved as: global_car_dataset.csv")