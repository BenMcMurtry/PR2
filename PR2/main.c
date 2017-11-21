//  PR2
//
//  Created by Ben McMurtry on 18/01/2017.
//  Copyright © 2016 Ben McMurtry. All rights reserved.


/*  
 This program simulates a 2D ferromagnet using the statistical mechanics approach of the Ising Model. An lattice of randomly distributed integer spins is generated, and the Metropolis algorithm is used to evolve the lattice in time until equilibrium is reached (which is defined to be when the Energy of the system is approximately constant). At equilibrium the macroscopic thermal properties of lattice energy, average magnetisation per spin, specific heat capacity and susceptibility are calculated. The lattice is evolved this way, throughout a range of temperatures and the macroscopic properties at equilibrium for each temperature are printed to file.
*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <errno.h>
#include <stdbool.h>


/* Definitions for lattice. */
#define D       (32)        // Number of spins along one edge of the spin lattice
#define UP      (1)         // Spin up
#define DOWN    (-1)        // Spin down
#define SEED time(NULL)     // Seed for srand()
#define LATTICE_N (D*D)     // Number of lattice points
#define INVERSE_N   (1.0 / LATTICE_N)   // 1/N since floating point multiplication is faster than division
#define NEAR_NEIGHBOURS   (4)   // Number of nearest neighbours for use with flip_probabilities array
#define S(x,y)  S[(D+(x)) % D][(D+(y)) % D]     // S access macro to simulate infinite lattice with periodic boundary conditions

/* Defintions for statistical model. */
#define MU0B    (0.0)       // External applied field.
#define K_B     (1.0)       // Boltzmann's constant set to 1 by unit conversion [from 1.38065e-23 Joules K^-1]
#define J       (1.0)       // Exchange energy set to 1 by unit conversion [from Joules]
#define TEMP_MIN    (0.0)   // Initial temperatures [units of J/K_B]
#define TEMP_MAX    (4.0)   // Final temperature [units of J/K_B]
#define TEMP_STEP   (0.1)   // Temperature increments [units of J/K_B]
#define EQBM_THRESHOLD (1.0e-4)     // Threshold proportional energy change value to check equilibrium
#define THERMAL_EQBM_TIMES (2.0)        // Number of equilibrium times allowed for thermal fluctuation measurements
#define TIME_STEPS_MAX (1e9)        // Maximum number of steps allowed to find equilibrium

/* Definitions for gnuplot. */
#define GNUPLOT_EXE    "/opt/local/bin/gnuplot" /* YOUR path for gnuplot here. */
#define GNUPLOT_SCRIPT "PR2script.txt"
#define GNUPLOT_DATA   "PR2data.txt"


/* Stores sums of Energy and Magnetism, equilibrium time, and a counter (number of stores) for thermal property calculations. */
struct eqbm_sum {
    double sum_E_sq, sum_E, sum_M_sq, sum_M;
    int counter, equilibrium_time;
};

/* Stores physical properties calculated by calculate_properties() ready to write to file. */
struct thermal_properties{
    double temperature, heat_capacity, susceptability, energy, magnetisation;
    int equilibrium_time;
};


/* Lattice functions. */
static void randomise_spins(int S[D][D]);
static void display_lattice(int S[D][D]);
static void run_through_temperature(int S[D][D], char *filename);
static void time_evolve_metrop(int S[D][D], double *flip_probabilities, struct eqbm_sum *p_eqbm_sum, double temp);

/* Statistics funtions. */
static void calc_flip_probabilities(double *flip_probabilities, double temp);
static bool equilibrium_check(double current_energy, double old_energy);

/* Property calculators. */
static void calculate_properties(struct eqbm_sum *p_eqbm_sum, struct thermal_properties *props, int S[D][D], double temp);
static double lattice_energy(int S[D][D]);
static double magnetisation_per_spin(int S[D][D]);
static double calculate_heat_capacity(struct eqbm_sum *p_eqbm_sum, double temp);
static double calculate_susceptibility(struct eqbm_sum *p_eqbm_sum, double temp);

/* Results functions. */
static void write_results(struct thermal_properties *props, FILE* datafile);
static int plot_data();

/* xfunctions - Error checking versions of common functions, which exit in the case of a failure. */
static FILE* xfopen(char *filename, char *option);



int main(int argc, char **argv) {
    printf("Welcome to the Ising Model Simulator!\n\n");
    srand((unsigned int) SEED);
    
    int S[D][D];
    randomise_spins(S);
    
    display_lattice(S);
    
    run_through_temperature(S, GNUPLOT_DATA);
    
    plot_data();

    return 0;
}

/* Initialise each spin in the lattice to either -1.0 or 1.0, randomly. */
static void randomise_spins(int S[D][D]) {
    for (int x = 0; x < D; x++ ) {
        for (int y = 0; y < D; y++ ) {      // rand is not perfectly uniform,
            S(x,y) = (rand() % 2) * 2 - 1;  // but since we are modding a small number, the bias will be of order 1 / RAND_MAX
        }
    }
}

/* Display the lattice to stdout, with UP as a "+" and DOWN as a " ". */
static void display_lattice(int S[D][D]) {
    for (int x = 0; x < D; x++ ) {
        for (int y = 0; y < D; y++ ) {
            if (S(x,y) == UP){printf("+");}
            else if (S(x,y) == DOWN) {printf(" ");}
        }
        printf("\n");
    }
    printf("\n");
}

/* Evolves the lattice through the desired temperature range. Writes results to file. */
static void run_through_temperature(int S[D][D], char *filename) {
    double temp;
    double flip_probabilities[NEAR_NEIGHBOURS + 1]; // At each temperature there are 5 possible sums of adjacent spins, and thus 5 flip probabilities.
    
    struct eqbm_sum eqbm_sum_s = {0};
    struct thermal_properties props;
    
    FILE *datafile = xfopen(filename, "w");
    fprintf(datafile, "#2D Ising Model. Lattice Size: %d, Eqbm. Threshold: %f, Thermal Equilibrium Times: %f\n", D, EQBM_THRESHOLD, THERMAL_EQBM_TIMES);
    fprintf(datafile, "#%-16s %-16s %-16s %-16s %-16s %-16s\n\n", "Temperature", "Energy", "Magnetisation", "Heat Capacity", "Susceptibility", "Equilibrium Time");
    
    // Loop down through temperatures
    for(temp = TEMP_MAX; temp > TEMP_MIN; temp -= TEMP_STEP) {
        calc_flip_probabilities(flip_probabilities, temp);      // Calculate the Boltzmann flip probabilities at current temperature
        time_evolve_metrop(S, flip_probabilities, &eqbm_sum_s, temp);    // Metropolis algorithm to equilibrium and allow fluctuations
        calculate_properties(&eqbm_sum_s, &props, S, temp);     // Calculate equilibrium properties for each temperature
        props.temperature = temp;
        write_results(&props, datafile);        // Write results to file
        eqbm_sum_s = (struct eqbm_sum){0};      // Reset the thermal properties to 0 at each new temperature
    }
    
    fclose(datafile);
}

/* Calculates the array of flip_probabilities for a given temperature.
 This speeds up the metropolis algorithm by allowing us to avoid calculating the exponential at every time step. */
static void calc_flip_probabilities(double *flip_probabilities, double temp) {
    double delta_E;
    //The sum of the adjacent spins can go from -NEAREST NEIGHBOURS to NEAREST NEIGHBOURS in steps of +2 (one DOWN to UP = +2).
    for(int i = 0; i < NEAR_NEIGHBOURS + 1; i++) {
        delta_E = 2.0 * J * 2.0*(i-2.0) + MU0B;
        flip_probabilities[i] = exp(-(delta_E / (K_B * temp)));
    }
}

/* Applies the Metropolis algorithm to evolve the system in time until equilibrium is reached at each temperature. */
static void time_evolve_metrop(int S[D][D], double *flip_probabilities, struct eqbm_sum *p_eqbm_sum, double temp) {

    double flip_prob;
    double energy_shift = 0;
    double mag_shift = 0; // Shift value used in calculating variance to prevent catastrophic cancellation
    
    bool equilbrium1 = false;       // Equilbrium will only be recognised if it is found twice in a row
    bool equilbrium2 = false;
    
    int time_step = 0;
    double energy = lattice_energy( S );
    double mag = magnetisation_per_spin( S );
    double old_energy = energy;         // Store current energy for equilibrium_check.
    
    while(time_step < TIME_STEPS_MAX) {     //Go through lattice sequentially to equilibrate faster
        for (int x = 0; x < D; x++) {
            for (int y = 0; y < D; y++) {
                time_step++;
                
                int sum_adj_spin = S(x-1,y) + S(x,y-1) + S(x+1,y) + S(x,y+1);
                double delta_E = S(x,y) * (2.0 * J * sum_adj_spin + MU0B);
                double delta_M = -2 * S(x,y) * INVERSE_N;
                
                // The next two lines calculate which value from the flip_probabilities array to use, by mapping from the sum_adj_spins.
                if(S(x,y) == DOWN) {flip_prob = 1 / flip_probabilities[((int)sum_adj_spin / 2) + 2];}
                else {flip_prob = flip_probabilities[((int)sum_adj_spin / 2) + 2];}
                
                double U = rand() / (double)RAND_MAX;           // U = a random [0,1] variate.
                if(U < flip_prob) {
                    S(x,y) = -S(x,y);
                    energy 	+= delta_E;     //Modify total energy and magnetisation if spin flipped.
                    mag 	+= delta_M;
                }
            }
        }
        
        if(!equilbrium1) {          // If not equilibrium check for equilibrium
            equilbrium1 = equilibrium_check(energy, old_energy);
            old_energy = energy;
            if(equilbrium1 && !equilbrium2) {
                equilbrium2 = true;     // Swap the true to equilbrium2 so we can contine assigning check to equilibrium1
                equilbrium1 = false;
            }
            else if(!equilbrium1 && equilbrium2) {
                equilbrium2 = false;    // One success followed by a fail, resets the success
            }
            else if(equilbrium1 && equilbrium2)	{        // If equilibrium is found twice in a row we store equilibrium time
                p_eqbm_sum->equilibrium_time = time_step;
                energy_shift = energy;                  // Energy and Magnetisation are stored in a shift for the variance calculation
                mag_shift = mag;
            }
        }
        if(equilbrium1 && equilbrium2) {        // If equilibrium store energy and magnetisation data
            p_eqbm_sum->sum_E_sq	+= ((energy - energy_shift) * (energy - energy_shift));
            p_eqbm_sum->sum_E	 	+= (energy - energy_shift);
            p_eqbm_sum->sum_M_sq 	+= ((mag - mag_shift) * (mag - mag_shift));
            p_eqbm_sum->sum_M       += (mag - mag_shift);
            p_eqbm_sum->counter++;
        }
        if(equilbrium1 && equilbrium2 && time_step > THERMAL_EQBM_TIMES * p_eqbm_sum->equilibrium_time) {
            printf("Equilibrium Found at Temp: %3g Steps: %d\n", temp, p_eqbm_sum->equilibrium_time);
            break;      // Break after a certain multiple (THERMAL_EQBM_TIMES) of equilibrium time
        }	
    }
    if(!(equilbrium1 && equilbrium2)) {
        printf("Failed to find Equilibrium. This is written in file as Equilibrium Time = 0 \n");
    }
}

/* Calculates the proportional change in energy and uses the eqbm_threshold option to check if equilibrium has reached. */
static bool equilibrium_check(double current_energy, double old_energy) {
    double Energy_change = fabs((old_energy - current_energy) / current_energy);     //Proportional change in energy
    if(Energy_change < EQBM_THRESHOLD) {return true;}
    else {return false;}
}

/* Calculates all the thermal properties for a given temperature, and modifies the props struct. */
static void calculate_properties(struct eqbm_sum *p_eqbm_sum, struct thermal_properties *props, int S[D][D], double temp) {
    props->energy           = lattice_energy(S);
    props->magnetisation	= magnetisation_per_spin(S);
    props->heat_capacity	= calculate_heat_capacity(p_eqbm_sum, temp);
    props->susceptability 	= calculate_susceptibility(p_eqbm_sum, temp);
    props->equilibrium_time = p_eqbm_sum->equilibrium_time;
}

/* Calculates and returns the total Energy, E_total, for the current lattice configuration. */
static double lattice_energy(int S[D][D]) {
    double E_total = 0.0;
    for (int x = 0; x < D; x++) {
        for (int y = 0; y < D; y++) {
            E_total -= S(x,y) * (J * (S(x+1,y) + S(x,y+1)) + MU0B);
        }
    }
    return E_total;
}

/* Calculates and returns the average Magnetisation per spin, for the current lattice configuration. */
static double magnetisation_per_spin(int S[D][D]) {
    double mag_tot = 0.0;
    for (int x = 0; x < D; x++) {
        for (int y = 0; y < D; y++) {
            mag_tot += S(x,y);
        }
    }
    return mag_tot / (double) (D*D);
}

/* Calculates the Heat Capacity of the lattice using the eqbm_sum struct. */
static double calculate_heat_capacity(struct eqbm_sum *p_eqbm_sum, double temp) {
    double time_avg_E_sq = p_eqbm_sum->sum_E_sq / (double) p_eqbm_sum->counter;
    double E_time_avg_sq = (p_eqbm_sum->sum_E / (double) p_eqbm_sum->counter) * (p_eqbm_sum->sum_E / (double) p_eqbm_sum->counter);
    return (1 / (K_B * temp * temp)) * (time_avg_E_sq - E_time_avg_sq);
}

/* Calculates the Susceptibility of the lattice using the eqbm_sum struct. */
static double calculate_susceptibility(struct eqbm_sum *p_eqbm_sum, double temp) {
    double time_avg_M_sq = p_eqbm_sum->sum_M_sq / (double) p_eqbm_sum->counter;
    double M_time_avg_sq = (p_eqbm_sum->sum_M / (double) p_eqbm_sum->counter) * (p_eqbm_sum->sum_M / (double) p_eqbm_sum->counter);
    return (1 / (K_B * temp)) * (time_avg_M_sq - M_time_avg_sq);
}

/* Same functionality as fopen, but exits if file opening fails. */
static FILE* xfopen(char *filename, char *option) {
    FILE *fp = fopen(filename, option);
    if(fp == NULL) {
        fprintf(stderr, "File %s failed to open.\n", filename);
        exit(EXIT_FAILURE);
    }
    return fp;
}

/* Writes properties to datafile. */
static void write_results(struct thermal_properties *props, FILE *datafile) {
    fprintf(datafile, "%-16f %-16e %-16e %-16f %-16f %-16d \n", props->temperature, props->energy, props->magnetisation, props->heat_capacity, props->susceptability, props->equilibrium_time);
}

/* Function to write a script for gnuplot to plot data. */
static int plot_data() {
    char command[PATH_MAX];
    
    FILE *script = fopen(GNUPLOT_SCRIPT, "w");
    if (!script) {return errno;}
    fprintf(script,
            "set multiplot layout 2,2\n"
            "set key off\n"
            "set title \"Ising Model\"\n"
            "set grid\n"
            "set xlabel \"Temperature[J/k_B]\"\n"
            "set ylabel \"Energy [J]\"\n"
            "plot \"PR2data.txt\" u 1:2 t \"Average Energy vs Temp\" w linespoints lw 0.5 pt 2 ps 0.7 lt 1\n"
            "set xlabel \"Temperature[J/k_B]\"\n"
            "set ylabel \"Magnetisation [m]\"\n"
            "plot \"PR2data.txt\" u 1:3 t \"Average Mag vs Temp\" w linespoints lw 0.5 pt 2 ps 0.7 lt 2\n"
            "set xlabel \"Temperature[J/k_B]\"\n"
            "set ylabel \"Heat Capacity [J / k_B^2]\"\n"
            "plot \"PR2data.txt\" u 1:4 t \"Heat Capacity vs Temp\" w linespoints lw 0.5 pt 2 ps 0.7 lt 3\n"
            "set xlabel \"Temperature[J/k_B]\"\n"
            "set ylabel \"Susceptibility\"\n"
            "plot \"PR2data.txt\" u 1:5 t \"Susceptibility vs Temp\" w linespoints lw 0.5 pt 2 ps 0.7 lt 4\n");
    fclose(script);
    
    snprintf(command, sizeof(command), "%s %s", GNUPLOT_EXE, GNUPLOT_SCRIPT );
    system( command );
    
    return(0);
}

