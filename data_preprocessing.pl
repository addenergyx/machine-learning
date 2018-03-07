#! /usr/bin/perl 
# Shebang line, this is the path to the perl binary

use strict; # Forces correct coding practises like declaring variables

# Catches and explains errors
use warnings;
use diagnostics;

use RNA;
use Text::CSV_XS;
use Getopt::Long;
use Text::CSV::Slurp;
use Try::Tiny;
use Log::Log4perl qw( :easy );
Log::Log4perl->easy_init( $INFO );

# This script is to prepare the dataset for the neural network. Andrew suggested other features that they don't currently record such as number of a specific nucleotide and g-c content. This code will take the sequence from each observation to get the features required.

# Inputs 
#   CSV file with 7 columns  
#   One obversation from the lab would be represented like this:

#   -GGTTCCAGAACCGGAGGACAAAGTACAAACGGCAGAAGCTGGAGGAGGAAGGGCCTGAGTCCGAGCAGAAGAAGAAGGGCTCCCATCACATCAACCGGTGGCGCATTGCCACGAAGCAGGCCAATGGGGAGGACATCGATGTCACCTCCAATGACTAGGGTGGGCAACCACAAACCCACGAGGGCAGAGTGCTGCTTGCTGCTGGCCAGGCCCCTGCGTGGGCCCAAGCTGGACTCTGGCCACTCCCTGGCCAGGCTTTGGGGAGGCCTGGAGTCATG	False	True	False	0	0	0	1684	34.9885726158

my $raw_data = 'Example_summary.csv'; 

# To allow the user to enter their own csv file and output file. CSV must be in the same schema as default
GetOptions(
	'sample=s' => \$raw_data,
	'output=s'=> \my $new_file,
);

# To differentiate the labs csv from the one used for the neural network 'NN' will be concatenated to the beginning of the filename unless stated otherwise using the 'output' flag
$new_file = 'Neural_network_' . $raw_data;

my $exit = 0;
my $total = 0;

# Checks if file already exists
LABEL: if (-e $new_file) {
    print "The output file '$new_file' already exists, do you wish to overwrite it? [Y/N] ";
	my $input = <STDIN>;	
	chomp $input;
		if ($input =~ /^n$/i) {
			die ERROR "The file '$new_file' exists, rename your output file using --output <filename>";
		} elsif ($input !~ /^y$/i) {
			$exit++;
			if ($exit == 3) { 
				ERROR print "Too many failed attempts\n";
				exit(0);
			}
			goto LABEL;
		}
}

INFO "Creating data from '$raw_data' into file '$new_file'";

my $csv = Text::CSV_XS->new() or die Text::CSV_XS->error_diag();
my @dataset;

open (my $data, '<', $raw_data) or die "Could not find or open '$raw_data'\n";

# Arrayref of column names 
my $headers = $csv->getline($data);

# Using q instead of "" as the latter appears in the csv and quotes are another common separated value 
my $new_features = [q/a_count/,q/c_count/,q/t_count/,q/g_count/,q/gc_content/,q/tga_count/,q/ttt_count/,q/minimum_free_energy_prediction/,q/ins_dels/];

# Although the amplicon is not needed for the neural network I am going to include it in this new csv as the lab may need it for reference. The neural network will ignore this column.
my @dataset_header =  $csv->column_names( @{ $headers}[0..6], @{$new_features}[0..7], @{$headers}[7,8], @{$new_features}[8] );
$csv->column_names( @{ $headers} );

while (my $observation = $csv->getline_hr($data)) {	
	if (scalar keys %{$observation} == 9){
		my $sequence = $observation->{'Aligned_Sequence'};
   		my @features = variables($sequence);

		#output column
		my $output = ($observation->{'n_deleted'}*-1) + $observation->{'n_inserted'};
		print "\nNumber of changes to aligned sequence \n$output\n";

		my $results = {

			Aligned_Sequence => $observation->{'Aligned_Sequence'},
			NHEJ => $observation->{'NHEJ'},
			UNMODIFIED => $observation->{'UNMODIFIED'},
			HDR => $observation->{'HDR'},
			n_deleted => $observation->{'n_deleted'},
			n_inserted => $observation->{'n_inserted'},
			n_mutated => $observation->{'n_mutated'},
			@{$new_features}[0] => $features[0],
			@{$new_features}[1] => $features[1],
			@{$new_features}[2] => $features[2],
			@{$new_features}[3] => $features[3],
			@{$new_features}[4] => $features[4],
			@{$new_features}[5] => $features[5],
			@{$new_features}[6] => $features[6],
			@{$new_features}[7] => $features[7],
			'#Reads' => $observation->{'#Reads'},
			'%Reads' => $observation->{'%Reads'},
			ins_dels => $output,

		};

		push (@dataset, $results);
		
	} else {
		
		die "Your CSV does not match intended format Please amend CSV";

	}
}

close $data;

# Putting the dataset in at once instead of adding in row by row to avoid memory leak by having the file open for as little as necessary
 
open (my $out, '>', $new_file) or die "Could not create file '$new_file': $!\n";
try {
	my $slurp = Text::CSV::Slurp->create(input => \@dataset, field_order => \@dataset_header);
	print $out $slurp;
} catch {
	warn "Caught error: $_";
};
close $out;

#Log file
#INFO "Number of records processed : $total";
#INFO "Number of bad records : ";

sub variables {
	my $sequence = shift;
	$total++;
# Output: Several features derived from the sequence 
#  A : 65 
#  C : 65 
#  T : 33 
#  G : 80 
#  G-C Content : 18
#  TTT Content : 1
#  Minimum free energy prediction: -91.5
	
	print "Sequence:\n$sequence\n\n";

# Nucleotide count

# Using () to change the the count from scalar to list context
#my $a_count = ($sequence =~ tr/a//i); <- doesn't work

	my $a_count = () = $sequence =~ /a/ig; 
	my $c_count = () = $sequence =~ /c/ig; 
	my $t_count = () = $sequence =~ /t/ig; 
	my $g_count = () = $sequence =~ /g/ig; 
	print "A : $a_count \nC : $a_count \nT : $t_count \nG : $g_count \n";

# G-C bonds are stronger then A-T bonds as they have an extra hydrogen bond, therefore making them harder to break
	my $gc_content = () = $sequence =~ /gc/ig;
	print "G-C Content : $gc_content\n"; 

# TGA is the stop codon, the ribosome would stop translation at this point
	my $tga_count = () = $sequence =~ /tga/ig;

# 3 or more T bases in a row can pervent polymerase from working properlly and therefore it can drop off
	my $triple_t_count = () = $sequence =~ /ttt/ig;
	print "TTT Content : $triple_t_count\n"; 

# Secondary structure: The lab uses RNAfold WebServer to get information on the secondary structure of the DNA/RNA sequence. ViennaRNA has a package I can install and use to get the features needed

	my ($structure, $mfe) = RNA::fold($sequence);
	print "Minimum free energy prediction: $mfe\n\n";

	my @features = ($a_count, $c_count, $t_count, $g_count, $gc_content, $tga_count, $triple_t_count, $mfe);
	print join ("\n", @features);
	
	return @features;
 
}



