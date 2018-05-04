#! /usr/bin/perl 
# Shebang line, this is the path to the perl binary

use strict; # Forces correct coding practises like declaring variables

# Catches and explains errors
use warnings;

use RNA;
use Text::CSV_XS;
use Getopt::Long;
use Text::CSV::Slurp;
use Try::Tiny;
use Log::Log4perl qw( :easy );
Log::Log4perl->easy_init( $INFO );
use Parallel::ForkManager;
use Text::CSV::Separator qw(get_separator);

# This script is to prepare the dataset for the neural network. Andrew suggested other features that they don't currently record such as number of a specific nucleotide and g-c content. This code will take the sequence from each observation to get the features required.

# Inputs 
#   CSV file with 7 columns  
#   One obversation from the lab would be represented like this:

#   -GGTTCCAGAACCGGAGGACAAAGTACAAACGGCAGAAGCTGGAGGAGGAAGGGCCTGAGTCCGAGCAGAAGAAGAAGGGCTCCCATCACATCAACCGGTGGCGCATTGCCACGAAGCAGGCCAATGGGGAGGACATCGATGTCACCTCCAATGACTAGGGTGGGCAACCACAAACCCACGAGGGCAGAGTGCTGCTTGCTGCTGGCCAGGCCCCTGCGTGGGCCCAAGCTGGACTCTGGCCACTCCCTGGCCAGGCTTTGGGGAGGCCTGGAGTCATG	False	True	False	0	0	0	1684	34.9885726158

#my @raw_data = 'csv/Example_summary.csv'; 
# To allow the user to enter their own csv file and output file. CSV must be in the same schema as default

# By default the program will run 1 process per cpu. This is because if parallel processes are competing for I/O bandwidth to keep reloading their data, they'll run slower than if they are ra sequentially.
my $procs = `nproc --all`;

GetOptions(
	'sample=s{1,}' => \my @raw_data,
	'output=s'	   => \my $new_file,
	'procs=i'	   => \$procs,
	'verbose'	   => \my $verbose,
	'force'		   => \my $force,
	'ins'		   => \my $ins,
	'dels'		   => \my $dels	
) or die("Command line argument error\n");

#my $total = 0;
#my $n_files = 0;
my $pm = Parallel::ForkManager->new($procs);

$pm->run_on_finish( sub {
	my ($pid, $exit_code, $ident, $exit_signal, $core_dump, $data_structure_reference) = @_;
#$DB::single=1;
		INFO "Process has finshed (pid:$pid)\n";
		INFO "$data_structure_reference->{observations} observations have been created in file '$data_structure_reference->{new_file}'\n"; 
});

if (scalar @raw_data == 0){@raw_data = @ARGV}

OUTER: foreach my $current_file (@raw_data) { 
	my $pid = $pm->start and next OUTER; # Forks and returns pid of child process
	#shift @raw_data;
	#$n_files++;

# To differentiate the labs csv from the one used for the neural network 'NN' will be concatenated to the beginning of the filename unless stated otherwise using the 'output' flag

### NEED TO FIX ###
#if (! defined $new_file) {
	if ($ins) {
		$new_file = 'Neural_network_insertions_' . $current_file;
	} elsif ($dels) {
		$new_file = 'Neural_network_deletions_' . $current_file;
	} else  {
		$new_file = 'Neural_network_' . $current_file;
	}
#}

# STDIN doesn't work properlly with forking
#my $exit = 0;
#if (! defined $force) {
# Checks if file already exists
#LABEL: if (-e $new_file) {
#    warn "The output file '$new_file' already exists, do you wish to overwrite it? [Y/N] ";
#	my $input = <STDIN>;	
#	chomp $input;
#		if ($input =~ /^n$/i) {
#			if ($n_files <= scalar @raw_data){
#				goto OUTER;
#			} else {
#				die ERROR "The file '$new_file' exists, rename your output file using --output <filename>";
#			}
#		} elsif ($input !~ /^y$/i) {
#			$exit++;
#			if ($exit == 3) { 
#				ERROR print "Too many failed attempts\n";
#				exit(0);
#			}
#			goto LABEL;
#		}
#}
#}
###

	my $sep_char = get_separator(
									path	=>	$current_file,
									include =>  ["\cI"],
									lucky	=>	1,
								 );

	if (! defined $sep_char){ die "Could not determine separator in CSV" }

	my $csv = Text::CSV_XS->new ({sep_char  =>  $sep_char}) or die Text::CSV_XS->error_diag();

	my @dataset;
	my $output_label;

	if ($ins) {
		$output_label = 'insertions';
	} elsif ($dels) {
		$output_label = 'deletions';
	} else {
		$output_label = 'ins/dels';
	}

	open (my $data, '<', $current_file) or die "Could not find or open '$current_file'\n";

# Arrayref of column names 
	my $headers = $csv->getline($data);

# Using q instead of "" as the latter appears in the csv and quotes are another common separated value 
# Putting features in an array is useful for Text::CSV::Slurp
	my $new_features = [q/a_count/,q/c_count/,q/t_count/,q/g_count/,q/gc_content/,q/tga_count/,q/ttt_count/,q/minimum_free_energy_prediction/,q/pam_count/,q/length/,q/frameshift/,$output_label];

# Although the amplicon is not needed for the neural network I am going to include it in this new csv as the lab may need it for reference. The neural network will ignore this column.
	my @dataset_header =  $csv->column_names( @{ $headers}[0..3], @{ $headers}[6], @{$new_features}[0..10], @{$headers}[7,8], @{$new_features}[11] );
$csv->column_names( @{ $headers} );

	LOOP: while (my $observation = $csv->getline_hr($data)) {	
		if (scalar keys %{$observation} == 9){

			INFO "Building data from '$current_file'";
			my $sequence = $observation->{'Aligned_Sequence'};
			my $insertions = $observation->{'n_inserted'};
			my $deletions = $observation->{'n_deleted'};
   		
			if ($dels) {
				if ($insertions > 0) {next LOOP};
			} elsif ($ins) {
				if ($deletions > 0) {next LOOP};
			}

			my @features = variables($sequence, $insertions, $deletions);

			my $results = {
				Aligned_Sequence => $observation->{'Aligned_Sequence'},
				NHEJ => $observation->{'NHEJ'},
				UNMODIFIED => $observation->{'UNMODIFIED'},
				HDR => $observation->{'HDR'},
				#n_deleted => $observation->{'n_deleted'},
				#n_inserted => $observation->{'n_inserted'},
				# Also removed n deleted/inserted from @dataset_header
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
				@{$new_features}[8] => $features[8],
				@{$new_features}[9] => $features[9],
				@{$new_features}[10] => $features[10],
				@{$new_features}[11] => $features[11],
			};

			push (@dataset, $results);
		
		} else {
		
			die "Your CSV does not match intended format Please amend CSV\n$_";

		}
	}

	close $data;

# Putting the dataset in at once instead of adding in row by row to avoid memory leak by having the file open for as little as necessary
 
	open (my $out, '>', $new_file) or die "Could not create file '$new_file': $!\n";
	try {
		my $slurp = Text::CSV::Slurp->create(input => \@dataset, field_order => \@dataset_header);
		print $out $slurp;
		INFO "Data stored into new neural network file '$new_file'";
	} catch {
		warn "Caught error: $_";
	};
	close $out;
	$pm->finish(0, {current_file => $current_file, new_file => $new_file, observations => scalar @dataset}); # Terminates child process
#undef ($new_file);
}
$pm->wait_all_children;

# Log file
#INFO "Number of records processed : $total";
#INFO "Number of bad records : ";

sub variables {
	my ($sequence, $insertions, $deletions) = @_;
	#$total++;
# Output: Several features derived from the sequence 
#  A : 65 
#  C : 65 
#  T : 33 
#  G : 80 
#  G-C Content : 18
#  TTT Content : 1
#  Minimum free energy prediction: -91.5

	#if (substr($sequence, 0, 1) eq '-'){$sequence =~ s/^-//}

# Nucleotide count

# Using () to change the the count from scalar to list context
#my $a_count = ($sequence =~ tr/a//i); <- doesn't work

	my $a_count = () = $sequence =~ /a/ig; 
	my $c_count = () = $sequence =~ /c/ig; 
	my $t_count = () = $sequence =~ /t/ig; 
	my $g_count = () = $sequence =~ /g/ig; 

# G-C bonds are stronger then A-T bonds as they have an extra hydrogen bond, therefore making them harder to break
	my $gc_content = () = $sequence =~ /(?=(gc)|(cg))/ig;

# TGA is the stop codon, the ribosome would stop translation at this point
	my $tga_count = () = $sequence =~ /tga/ig;

# 3 or more T bases in a row can pervent polymerase from working properlly and therefore it can drop off
	my $triple_t_count = () = $sequence =~ /ttt/ig;

# Secondary structure: The lab uses RNAfold WebServer to get information on the secondary structure of the DNA/RNA sequence. ViennaRNA has a package I can install and use to get the features needed
	my ($structure, $mfe) = RNA::fold($sequence);

# PAM count:  The PAM is where CAS-9 pauses to make cuts. Currently the only PAM used is NGG however this may change in the future
	my $PAM_count = () = $sequence =~ /.gg/ig;

# Length of sequence
	my $length = length($sequence);

# Frameshift: A frameshift mutation occurs by the number of in/dels not being divisible by 3. As a result the translation of the sequence produces a different amino acid and therefore protein.
	my ($in_frameshift, $del_frameshift, $frameshift);	
    
	#Should do frameshift for each cut site not total, need ref seq to do this for insertions
	if ($dels || (! defined $dels && ! defined $ins)) {  
		if ($deletions % 3 == 0){
			$del_frameshift = 'False';
		} else {
			$del_frameshift = 'True';
		}	
	}

	if ($ins || (! defined $dels && ! defined $ins)) {
		if ($insertions % 3 == 0){
			$in_frameshift = 'False';
		} else {
			$in_frameshift = 'True';
		}
	}

# Perl and python don't have switch-case statements, perl has a switch module but it is buggy and has been removed from core.

	if (! defined $in_frameshift){
		$frameshift = $del_frameshift;
	} elsif (! defined $del_frameshift) {
		$frameshift = $in_frameshift;
	} else {
		if ($in_frameshift eq 'False' && $del_frameshift eq 'False'){
			$frameshift = 'False';	
		} else {
	    	#$in_frameshift eq 'true' || $del_frameshift eq 'true'
			$frameshift = 'True';
		}
	}

# Output insertion/deletion
	my $output;
		if ($dels) {
			$output = $deletions;
		} elsif ($ins) {
			$output = $insertions;
		} else {
			$output = ($deletions*-1) + $insertions;
			#print "\nNumber of changes to aligned sequence \n$output\n";
		}
	
	my @features = ($a_count, $c_count, $t_count, $g_count, $gc_content, $tga_count, $triple_t_count, $mfe, $PAM_count, $length, $frameshift, $output);
	#print join ("\n", @features);

	if ($verbose) {
		print "Sequence: $sequence\n\n";
		print "A : $a_count \nC : $a_count \nT : $t_count \nG : $g_count \n";
		print "G-C Content : $gc_content\n"; 
		print "TTT Content : $triple_t_count\n"; 
		print "Minimum free energy prediction: $mfe\n";
		print "Frameshift: $frameshift\n";
		print "Length of sequence: $length\n"; 
		print "PAM count: $PAM_count\n";
		print "Frameshift: $frameshift\n";
		print "Output: $output\n\n";
	}
	return @features;
}


