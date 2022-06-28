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

-- Creating tables for apple
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
SELECT * FROM apple;

-- Creating tables for kellogg
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

-- Creating tables for chrobinson
CREATE TABLE chrobinson (
     date DATE NOT NULL,
	 opening FLOAT NOT NULL,
     high FLOAT NOT NULL,
     low FLOAT NOT NULL,
	 closing FLOAT NOT NULL,
	 adj_closing FLOAT NOT NULL,
	 volume INT NOT NULL,
	 PRIMARY KEY (date),
     UNIQUE (date)
);

DROP TABLE chrobinson;
SELECT * FROM chrobinson;

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

-- Creating tables for exxon
CREATE TABLE exxon (
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
SELECT * FROM exxon;


-- Creating tables for conoco phillips
CREATE TABLE conoco (
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
SELECT * FROM conoco;

-- create Energy stocks table
SELECT date, opening, volume
INTO energy_stocks
FROM exxon;

SELECT * FROM energy_stocks;

-- join conoco to exxon
SELECT exxon.date,
		exxon.opening,
		exxon.volume
FROM exxon
LEFT JOIN conoco
ON exxon.date = conoco.date;
