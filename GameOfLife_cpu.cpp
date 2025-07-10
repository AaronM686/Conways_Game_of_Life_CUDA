////////////////////////////////////////////////////////////////////////////////
//
// CUDA implementation of "Conway's Game of Life" cellular automaton.
//   https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
//
// This is a coding skills demonstration created by Aaron Mosher.
// https://github.com/AaronM686
//
// Makefile and boilerplate support code is based on Nvidia samples "Template"
// You will need the Samples directory to compile this, since
// I rely on several helper-functions they provide to streamline the code.
// 
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeTick(unsigned int *reference, unsigned int *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeTick(unsigned int *reference, unsigned int *idata, const unsigned int len)
{
	// printf("computeTick: not implemented yet\n");
}

