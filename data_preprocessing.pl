#! /usr/bin/perl
# Shebang line, this is the path to the perl binary


use strict; # Forces correct coding practises like declaring variables

#Catches and explains errors
use warnings;
use diagnostics;

use RNA;
use Text::CSV_XS;
use Getopt::Long;
use Text::CSV::Slurp;

# This script is to prepare the dataset for the neural network. Andrew suggested other features that they don't currently record such as number of a specific nucleotide and g-c content. This code will take the sequence from each observation to get the features required.

# Inputs 
#   CSV file with 7 columns  
#   One obversation from the lab would be represented like this:

#   -GGTTCCAGAACCGGAGGACAAAGTACAAACGGCAGAAGCTGGAGGAGGAAGGGCCTGAGTCCGAGCAGAAGAAGAAGGGCTCCCATCACATCAACCGGTGGCGCATTGCCACGAAGCAGGCCAATGGGGAGGACATCGATGTCACCTCCAATGACTAGGGTGGGCAACCACAAACCCACGAGGGCAGAGTGCTGCTTGCTGCTGGCCAGGCCCCTGCGTGGGCCCAAGCTGGACTCTGGCCACTCCCTGGCCAGGCTTTGGGGAGGCCTGGAGTCATG	False	True	False	0	0	0	1684	34.9885726158

my $raw_data = 'Example_summary.csv';

# To differentiate the labs csv from the one used for the neural network 'NN' will be concatenated to the beginning of the filename unless stated otherwise using the 'outfile' flag 
my $new_file =  'Neural_network_' . $raw_data;

my $csv = Text::CSV_XS->new() or die Text::CSV_XS->error_diag();
my @dataset;

# To allow the user to enter their own csv file and output file. CSV must be in the same schema as default
GetOptions(
	'file=s' => \$raw_data,
	'outfile=s'=> \$new_file
);

open (my $data, '<', $raw_data) or die "Could not find or open '$raw_data'\n";

# Arrayref of column names 
my $headers = $csv->getline($data);

# Using qw instead of "" as the latter appears in the csv and quotes are another common separated value 
my $new_features = [q(a_count),qq/c_count/,q/t_count/,q/g_count/,q/gc_content/,q/ttt_count/,q/minimum_free_energy_prediction/];

# Although the amplicon is not needed for the neural network I am going to include it in this new csv as the lab may need it for reference. The neural network will ignore this column.
my @dataset_header =  $csv->column_names( @{ $headers}[0..6], @{$new_features}, @{$headers}[7,8] );
$csv->column_names( @{ $headers} );
	
	while (my $observation = $csv->getline_hr($data)) {	
#	if ($csv->parse($observation)){
		my $sequence = $observation->{'Aligned_Sequence'};
   		my @features = variables($sequence);

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
			'#Reads' => $observation->{'#Reads'},
			'%Reads' => $observation->{'%Reads'},

		};

push (@dataset, $results);
		
#	} else {
		
#		warn "Line could not be parsed: $observation\nPlease amend CSV";

#	}
}

# Putting the dataset in at once instead of adding in row by row to avoid memory like by having the file open for as little as necessary
 
open (my $out, '>', $new_file) or die "Could not create file '$new_file': $!\n";
my $slurp = Text::CSV::Slurp->create(input => \@dataset, field_order => \@dataset_header);
print $out $slurp;

close $data;
close $out;

=pod
OUTER: while (my $observation = <$data>){

	chomp $observation;

	if ($csv->parse($observation)){

		my @columns = $csv->fields();
		if ($columns[0] eq 'Aligned_Sequence'){
			my @column_names = @columns;
			next OUTER;
		}
		my $sequence = $columns[0];
		my @features = variables($sequence);
		
#At this point @features should contain the new features that need to be added to the dataset, the sequence itself isn't needed for the neural network

		my @data_set = (@columns[1,2,3,4,5,6],@features,@columns[7,8]);
		print "\n\n@data_set\n\n";

#ISSUE - @features should be before the last (output) column. The neural network can still work but to match convension the output should be the last column. Probably a much neater way to do it, [1-6] & [1:6] did not work 
#FIX - In perl use [1..6]

	} else {
		
		warn "Line could not be parsed: $observation\nPlease amend CSV";

	}


}
=cut

sub variables {
	my $sequence = shift;

# Output: Several features derived from the sequence 
#  A : 65 
#  C : 65 
#  T : 33 
#  G : 80 
#  G-C Content : 18
#  TTT Content : 1
#  Minimum free energy prediction: -91.5

	print "\n\nSequence:\n$sequence\n\n";

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

# 3 or more T bases in a row can pervent polymerase from working properlly and therefore it can drop off
	my $triple_t_count = () = $sequence =~ /ttt/ig;
	print "TTT Content : $triple_t_count\n"; 

# Secondary structure: The lab uses RNAfold WebServer to get information on the secondary structure of the DNA/RNA sequence. ViennaRNA has a package I can install and use to get the features needed

	my ($structure, $mfe) = RNA::fold($sequence);
	print "Minimum free energy prediction: $mfe\n\n";

	my @features = ($a_count, $c_count, $t_count, $g_count, $gc_content, $triple_t_count, $mfe);
	print join ("\n", @features);
	
	return @features;
 
}



