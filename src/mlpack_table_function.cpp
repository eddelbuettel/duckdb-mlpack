
#include "mlpack_table_function.hpp"

namespace duckdb {

static bool verbose = false;

// (Sample) Table code below -- for sample export of 5 columns with one simple scalar as param

unique_ptr<FunctionData> MlpackTableBind(ClientContext &context, TableFunctionBindInput &input,
                                         vector<LogicalType> &return_types, vector<string> &names) {

	auto resdata = make_uniq<MlpackTableData>(); // 'resdata' for result data i.e. outgoing

	if (verbose)
		std::cout << "MlpackTableBind\n";
	if (verbose)
		std::cout << "  old value=" << resdata->value << std::endl;
	resdata->value = input.inputs[0].GetValue<int>();
	if (verbose)
		std::cout << "  setting value=" << resdata->value << std::endl;

	names = {"col_1", "col_2", "col_3", "col_4", "col_5"};
	return_types = {LogicalType::SMALLINT, LogicalType::INTEGER, LogicalType::BIGINT, LogicalType::FLOAT,
	                LogicalType::DOUBLE};

	return std::move(resdata);
}

void MlpackTableFunction(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {

	if (verbose)
		std::cout << "MlpackTableFunction\n";
	auto &resdata = const_cast<MlpackTableData &>(data_p.bind_data->Cast<MlpackTableData>());

	// if we have been called, return nothing
	if (resdata.data_returned) {
		output.SetCardinality(0);
		return;
	}
	if (verbose)
		std::cout << "  seeing value=" << resdata.value << std::endl;

	idx_t chunk_size = 3; // arbitrary

	output.SetCardinality(chunk_size);

	int32_t val = resdata.value;

	for (idx_t i = 0; i < chunk_size; i++) {
		output.data[0].SetValue(i, val++);
		output.data[1].SetValue(i, val++);
		output.data[2].SetValue(i, val++);
		output.data[3].SetValue(i, val++);
		output.data[4].SetValue(i, val++);
	}

	// mark that we have been called
	resdata.data_returned = true;
}

} // namespace duckdb
