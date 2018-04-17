#ifndef INCLUDED_LDA_IO_HPP
#define INCLUDED_LDA_IO_HPP

#include <istream>
#include <ostream>

#include "lda.hpp"


// Saves a trained latent_dirichlet_allocation object to a textual stream.
void save_lda(std::ostream& output, latent_dirichlet_allocation const& lda);

// Loads a latent_dirichlet_allocation object from a textual stream.
latent_dirichlet_allocation load_lda(std::istream& input);


#endif
