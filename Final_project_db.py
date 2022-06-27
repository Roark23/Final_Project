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

-- Creating tables for chrobinson
CREATE TABLE chrobinson (
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