#! /usr/bin/perl 

use strict;
use warnings;
use RNA;
use Text::CSV_XS;
use Getopt::Long;

# This script is to prepare the dataset for the neural network. Andrew suggested other features that they don't currently record such as number of a specific nucleotide and g-c content. This code will take the sequence from each observation to get the features required.

# Inputs: csv file with .. columns:  
#   One obversation from the lab would be represented like this 

my $raw_data = 'Example_summary.csv';
my $csv = Text::CSV_XS->new();

# To allow user to enter their own csv file. Must be in the same schema as default
GetOptions(
	'file=s' => \$raw_data
);

open (my $data, '<', $raw_data) or die "Could not find or open '$raw_data'\n";

OUTER: while (my $observation = <$data>){

	chomp $observation;

	if ($csv->parse($observation)){

		my @columns = $csv->fields();
		if ($columns[0] eq 'Aligned_Sequence'){next OUTER};
		my $sequence = $columns[0];
		my @feature = variables($sequence);

	} else {
		
		warn "Line could not be parsed: $observation\nPlease amend CSV";

	}


}

sub variables {

	my $sequence = shift;
# Output: Several features derived from the sequence 

	print "\nSequence:\n$sequence\n\n";
	$DB::single=1;
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

# Later should look into using a hash containing these variables for each sequence
# my $string = q<ABCDABDIDAOFOOFAA>;
# my %count;
# for (split(//, $string)){$count{$_}++;}
# print(q<Matches >, $count{A}, "\n");

# Secondary structure: The lab uses RNAfold WebServer to get information on the secondary structure of the DNA/RNA sequence. ViennaRNA has a package I can install and use to get the features needed

	my ($structure, $mfe) = RNA::fold($sequence);
	print "Minimum free energy prediction: $mfe\n\n";

	my @features = ($a_count, $c_count, $t_count, $g_count, $gc_content, $triple_t_count, $mfe);
	print join ("\n", @features);
	
$DB::single=1;
	return @features;
 
}



