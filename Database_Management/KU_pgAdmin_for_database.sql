-- Creating tables for stock index
CREATE TABLE spstock (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);

--DROP TABLE spstock;
SELECT * FROM spstock;

-- Creating table for apple
CREATE TABLE apple (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);

DROP TABLE apple;
SELECT * FROM apple;

SELECT
   date,	
   closing AS aaple
FROM aaple;   

-- Creating tables for technology
CREATE TABLE tech_sector (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);

SELECT * FROM tech_sector;
DROP TABLE tech_sector;

-- Join tech_sector and apple and spstock closing prices
SELECT tech_sector.date,
		tech_sector.closing AS tech_sector,
		apple.closing AS aaple,
		spstock.closing AS spstock
INTO tech_sector_export		
FROM tech_sector
LEFT JOIN apple ON tech_sector.date = apple.date
LEFT JOIN spstock ON tech_sector.date = spstock.date
;

-- --------------------------------------------------------------------------------------

-- Creating table for kellogg
CREATE TABLE kellogg (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);
SELECT * FROM kellogg;

-- Creat table for Consumer Staples Sector
CREATE TABLE consumer_staples (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);
SELECT * FROM consumer_staples;

-- Join consumer_staples_sector and kellogg and spstock closing prices
SELECT consumer_staples.date,
		consumer_staples.closing AS tech_sector,
		kellogg.closing AS kellogg,
		spstock.closing AS spstock
INTO consumer_staples_sector_export		
FROM consumer_staples
LEFT JOIN kellogg ON consumer_staples.date = kellogg.date
LEFT JOIN spstock ON consumer_staples.date = spstock.date
;

SELECT * FROM consumer_staples_sector_export;

COPY consumer_staples_sector_export TO '/Users/kuric/KU_consumer_staples_sector_export.csv' DELIMITER ',' CSV HEADER;

-- --------------------------------------------------------------------------------------------

-- Creating tables for nike
CREATE TABLE nike (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);
SELECT * FROM nike;

-- Creat table for Consumer Discretionary Sector
CREATE TABLE consumer_discretionary (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);
SELECT * FROM consumer_discretionary;

-- Join consumer_staples_sector and kellogg and spstock closing prices
SELECT consumer_discretionary.date,
		consumer_discretionary.closing AS tech_sector,
		nike.closing AS nike,
		spstock.closing AS spstock
INTO consumer_discretionary_sector_export		
FROM consumer_discretionary
LEFT JOIN nike ON consumer_discretionary.date = nike.date
LEFT JOIN spstock ON consumer_discretionary.date = spstock.date
;

SELECT * FROM consumer_discretionary_sector_export;

-- --------------------------------------------------------------------------------

-- Creating tables for chrobinson
CREATE TABLE chrobinson (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 volume INT NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);

DROP TABLE chrobinson;
SELECT * FROM chrobinson;

-- Creat table for Consumer Discretionary Sector
CREATE TABLE industrials (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);
SELECT * FROM industrials;
DROP TABLE industrials;

-- Join consumer_staples_sector and kellogg and spstock closing prices
SELECT industrials.date,
		industrials.closing AS tech_sector,
		chrobinson.closing AS chrobinson,
		spstock.closing AS spstock
INTO industrials_sector_export		
FROM industrials
LEFT JOIN chrobinson ON industrials.date = chrobinson.date
LEFT JOIN spstock ON industrials.date = spstock.date
;

SELECT * FROM industrials_sector_export;

-- ----------------------------------------------------------------------------

-- Creating tables for occidental petroleum
CREATE TABLE oxy (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);

SELECT * FROM oxy;
DROP TABLE oxy;


-- Creat table for Consumer Discretionary Sector
CREATE TABLE energy (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume DECIMAL NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);
SELECT * FROM energy;
DROP TABLE energy;

-- Join energy_sector and kellogg and spstock closing prices
SELECT energy.date,
		energy.closing AS tech_sector,
		oxy.closing AS oxy,
		spstock.closing AS spstock
INTO energy_sector_export		
FROM energy
LEFT JOIN oxy ON energy.date = oxy.date
LEFT JOIN spstock ON energy.date = spstock.date
;

SELECT * FROM energy_sector_export;
DROP TABLE energy_sector_export;
