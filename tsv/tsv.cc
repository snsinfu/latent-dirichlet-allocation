#include <algorithm>
#include <cctype>
#include <cstddef>
#include <istream>
#include <ostream>
#include <iterator>
#include <string>
#include <utility>

#include <xtensor/xtensor.hpp>

#include "tsv.hpp"


template<typename OutputIterator>
std::size_t parse_tsv_row(std::string const& row, OutputIterator output)
{
    std::size_t item_count = 0;

    auto const is_delim = [](char ch) {
        return std::isblank(ch);
    };
    auto token_begin = std::find_if_not(row.begin(), row.end(), is_delim);

    while (token_begin != row.end()) {
        auto const token_end = std::find_if(token_begin, row.end(), is_delim);
        *output++ = std::stod(std::string{token_begin, token_end});
        item_count++;
        token_begin = std::find_if_not(token_end, row.end(), is_delim);
    }

    return item_count;
}

tsv_tensor load_tsv(std::istream& input)
{
    tsv_tensor::container_type values;
    std::size_t row_count = 0;
    std::size_t col_count = 0;

    for (std::string line; std::getline(input, line); ) {
        col_count = parse_tsv_row(line, std::back_inserter(values));
        row_count++;
    }

    return tsv_tensor{std::move(values), {row_count, col_count}, {}};
}

void save_tsv(std::ostream& output, xt::xtensor<double, 2> const& tensor)
{
    std::size_t const row_count = tensor.shape()[0];
    std::size_t const col_count = tensor.shape()[1];

    for (std::size_t row = 0; row < row_count; ++row) {
        for (std::size_t col = 0; col < col_count; ++col) {
            output << tensor(row, col)
                   << (col == col_count - 1 ? '\n' : '\t');
        }
    }
}
