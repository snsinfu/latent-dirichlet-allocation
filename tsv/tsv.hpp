#ifndef INCLUDED_TSV_HPP
#define INCLUDED_TSV_HPP

#include <istream>
#include <ostream>
#include <vector>

#include <xtensor/xtensor.hpp>


// xtensor container type used to store tsv contents.
using tsv_tensor = xt::xtensor_container<std::vector<double>, 2, xt::layout_type::row_major>;

// Loads TSV into a two-dimensional tensor.
tsv_tensor load_tsv(std::istream& input);

// Saves a two-dimensional tensor into a TSV file.
void save_tsv(std::ostream& output, xt::xtensor<double, 2> const& tensor);


#endif
