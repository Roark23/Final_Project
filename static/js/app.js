// import the data from data.js
const tableData = data;

// get table references
var tbody = d3.select("tbody");

//Create a function and pass the argument data pass in the data from data.js
function buildTable(inputdata) {
  // First, clear out any existing data - standard step.
  tbody.html("");

  // Next, loop through each object in the data
  // and append a row and cells for each value in the row
  inputdata.forEach((dataRow) => {
    // Append a row to the table body.
    let row = tbody.append("tr");

    // Loop through each field in the dataRow and add
    // each value as a table cell
    Object.values(dataRow).forEach((val) => {
      let cell = row.append("td");
      cell.text(val);
    });
  });
}

// 1. Create a variable to keep track of all the filters as an object.
var filters = {}

// 3. Use this function to update the filters. this
function updateFilters() {

    // 4a. Save the element that was changed as a variable.
    let changedElement = d3.select(this);

    // 4b. Save the value that was changed as a variable.
    let elementValue = changedElement.property("value");
    console.log(elementValue);

    // 4c. Save the id of the filter that was changed as a variable.
    let filterId = changedElement.attr("id");
    console.log(filterId);
  
    // 5. If a filter value was entered then add that filterId and value
    if(elementValue) {
        filters[filterId] = elementValue;
    }
    else {
    delete filters[filterId];
    }

    // 6. Call function to apply all filters and rebuild the table
    filterTable();
}
  
  // 7. Use this function to filter the table when data is entered.
  function filterTable() {

    // 8. Set the filtered data to the tableData.
    var filteredTable = tableData;
  
    // 9. Loop through all of the filters and keep any data that
    // matches the filter values.
    let filterDataOnly = Object.entries(filters);
    for (let [idKey, elValue] of filterDataOnly) {
      filteredTable = filteredTable.filter(row => row[idKey] === elValue)
      console.log(idKey, elValue)
    };
  
    // 10. Finally, rebuild the table using the filtered data
    buildTable(filteredTable);
  }
  
  // 2. Attach an event to listen for changes to each filter
  d3.selectAll("input").on("change", updateFilters);
  
  // Build the table when the page loads
  buildTable(tableData);