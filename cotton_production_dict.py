# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 08:37:29 2023

@author: townsend.lansburg
"""




seasons = {'United States of America': {'start_day': 135, 'end_day': 282},
 'Brazil': {'start_day': 360, 'end_day': 160}, 
 'Colombia': {'start_day': 246, 'end_day': 343},
 'Argentina': {'start_day': 342, 'end_day': 140}, 
 'Spain': {'start_day': 122, 'end_day': 252},
 'Iran': {'start_day': 131, 'end_day': 252},
 'Iraq': {'start_day': 91, 'end_day': 237},
 'Syria': {'start_day': 121, 'end_day': 231}, 
 'Turkey': {'start_day': 136, 'end_day': 260}, 
 'Kyrgyzstan': {'start_day': 133, 'end_day': 263},
 'Tajikistan': {'start_day': 133, 'end_day': 268}, 
 'Turkmenistan': {'start_day': 139, 'end_day': 268},
 'Uzbekistan': {'start_day': 137, 'end_day': 249},
 'China': {'start_day': 158, 'end_day': 252}, 
 'Burma': {'start_day': 112, 'end_day': 255},
 'Australia': {'start_day': 341, 'end_day': 134},
 'Bangladesh': {'start_day': 219, 'end_day': 343}, 
 'India': {'start_day': 188, 'end_day': 266}, 
 'Pakistan': {'start_day': 158, 'end_day': 227},
 'Egypt': {'start_day': 117, 'end_day': 271},
 'Madagascar ': {'start_day': 330, 'end_day': 102}, 
 'Malawi': {'start_day': 318, 'end_day': 129}, 
 'Mozambique': {'start_day': 350, 'end_day': 132},
 'South Africa': {'start_day': 341, 'end_day': 129},
 'Tanzania': {'start_day': 5, 'end_day': 177},
 'Zambia': {'start_day': 318, 'end_day': 220},
"Bolivia": {"start_day": 314, "end_day": 113},
"Paraguay": {"start_day": 330, "end_day": 112},
 'Zimbabwe': {'start_day': 318, 'end_day': 126},
 'Ethiopia': {'start_day': 175, 'end_day': 289}, 
 'Kenya': {'start_day': 135, 'end_day': 300}, 
 'Sudan': {'start_day': 196, 'end_day': 347},
 'Uganda': {'start_day': 135, 'end_day': 330}, 
 'Benin': {'start_day': 188, 'end_day': 318},
 'Burkina Faso': {'start_day': 188, 'end_day': 318}, 
 'Chad': {'start_day': 188, 'end_day': 310}, 
 'Cameroon': {'start_day': 188, 'end_day': 318}, 
 'Central Africa Republic': {'start_day': 188, 'end_day': 318}, 
 "Cote d'Ivoire": {'start_day': 188, 'end_day': 318}, 
 'Ghana': {'start_day': 188, 'end_day': 318}, 
 'Guinea': {'start_day': 188, 'end_day': 318}, 
 'Mali': {'start_day': 196, 'end_day': 322}, 
 'Nigeria': {'start_day': 188, 'end_day': 318}, 
 'Niger': {'start_day': 196, 'end_day': 337},
 'Senegal': {'start_day': 196, 'end_day': 337},
 'Togo': {'start_day': 188, 'end_day': 318},
 'Azerbaijan': {'start_day': 112, 'end_day': 249},
 'Mexico': {'start_day': 120, 'end_day': 365},
  'Somalia': {'start_day': 41, 'end_day': 246}}







state_production = {
 'China': {'Xinjiang': 28.0,
  'Shandong': 12.0,
  'Henan': 8.0,
  'Hebei': 6.0,
  'Jiangsu': 6.0,
  'Anhui': 5.0,
  'Hubei': 5.0,
  'Sichuan': 4.0,
  'Shaanxi': 4.0,
  'Hunan': 4.0,
  'Jiangxi': 3.0,
  'Gansu': 2.0,
  'Yunnan': 2.0,
  'Inner Mongolia': 2.0,
  'Guangdong': 1.0,
  'Hainan': 1.0,
  'Guizhou': 1.0,
  'Fujian': 1.0,
  'Heilongjiang': 1.0,
  'Other': 2.0},
 'India': {'Gujarat': 26.0,
  'Maharashtra': 24.0,
  'Telangana': 15.0,
  'Karnataka': 10.0,
  'Haryana': 8.0,
  'Rajasthan': 5.0,
  'Punjab': 4.0,
  'Andhra Pradesh': 3.0,
  'Tamil Nadu': 2.0,
  'Madhya Pradesh': 2.0,
  'Other': 1.0},
 'United States of America': {'Texas': 50.0,
  'Georgia': 14.0,
  'Mississippi': 7.0,
  'Arkansas': 6.0,
  'California': 5.0,
  'Alabama': 4.0,
  'Missouri': 4.0,
  'North Carolina': 3.0,
  'Arizona': 2.0,
  'Louisiana': 2.0,
  'South Carolina': 2.0,
  'Tennessee': 2.0,
  'Other': 1.0},
 'Pakistan': {'Punjab': 73.0,
  'Sindh': 24.0,
  'Balochistan': 2.0,
  'Khyber Pakhtunkhwa': 1.0},
 'Brazil': {'Mato Grosso': 70.0,
  'Bahia': 14.0,
  'Goias': 5.0,
  'Minas Gerais': 4.0,
  'Maranhao': 3.0,
  'Sao Paulo': 2.0,
  'Other': 2.0},
 'Uzbekistan': {'Tashkent': 30.0,
  'Andijan': 16.0,
  'Sirdaryo': 11.0,
  'Kashkadarya': 9.0,
  'Surxondaryo': 9.0,
  'Jizzakh': 7.0,
  'Fergana': 7.0,
  'Namangan': 6.0,
  'Bukhara': 4.0,
  'Karakalpakstan': 1.0},
 'Australia': {'Queensland': 42.0, 'New South Wales': 38.0, 'Other': 20.0},
 'Turkmenistan': {'Ahal': 35.0,
  'Mary': 24.0,
  'Lebap': 22.0,
  'Balkan': 11.0,
  'Dasoguz': 7.0,
  'Other': 1.0},
 'Argentina': {'Chaco': 30.0,
  'Santiago del Estero': 26.0,
  'Santa Fe': 23.0,
  'Formosa': 7.0,
  'Salta': 5.0,
  'Other': 9.0},

 'Benin': {'Atakora': 16.0, 'Borgou': 15.0, 'Other': 69.0},
 
  'Uruguay':{
     'Artigas':8,
     'Cerro Largo':8,
     'Colonia':2,
     'Durazno':8,
     'Flores':8,
     'Florida':7,
     'Lavalleja':8,
     'Paysandu':8,
     'Rio Negro':7,
     'Rivera':7,
     'Salto':7,
     'Soriano':7,
     'Tacuarembo':7,
     'Treinta y Tres':7,
 },
 'Paraguay':{
     'Alto Parana':8,
     'Amambay':8,
     'Caaguazu':8,
     'Caazapa':8,
     'Canindeyu':8,
     'Concepcion':7,
     'Cordillera':8,
     'Guaira':8,
     'Itapua':8,
     'Misiones':8,
     'Neembucu':7,
     'Paraguari':7,
     'San Pedro':7,
 },
 'Bolivia':{
     'Beni':50,
     'Santa Cruz':50,}
     
     }